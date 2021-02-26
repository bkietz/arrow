// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/dataset/scanner.h"

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>

#include "arrow/compute/api_scalar.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/dataset_internal.h"
#include "arrow/dataset/scanner_internal.h"
#include "arrow/table.h"
#include "arrow/util/iterator.h"
#include "arrow/util/logging.h"
#include "arrow/util/task_group.h"
#include "arrow/util/thread_pool.h"

namespace arrow {
namespace dataset {

std::vector<std::string> ScanOptions::MaterializedFields() const {
  std::vector<std::string> fields;

  for (const Expression* expr : {&filter, &projection}) {
    for (const FieldRef& ref : FieldsInExpression(*expr)) {
      DCHECK(ref.name());
      fields.push_back(*ref.name());
    }
  }

  return fields;
}

Result<RecordBatchIterator> InMemoryScanTask::Execute() {
  return MakeVectorIterator(record_batches_);
}

Result<FragmentIterator> Scanner::GetFragments() {
  if (fragment_ != nullptr) {
    return MakeVectorIterator(FragmentVector{fragment_});
  }

  // Transform Datasets in a flat Iterator<Fragment>. This
  // iterator is lazily constructed, i.e. Dataset::GetFragments is
  // not invoked until a Fragment is requested.
  return GetFragmentsFromDatasets({dataset_}, scan_options_->filter);
}

Result<ScanTaskIterator> Scanner::Scan() {
  // Transforms Iterator<Fragment> into a unified
  // Iterator<ScanTask>. The first Iterator::Next invocation is going to do
  // all the work of unwinding the chained iterators.
  ARROW_ASSIGN_OR_RAISE(auto fragment_it, GetFragments());
  return GetScanTaskIterator(std::move(fragment_it), scan_options_, scan_context_);
}

Result<ScanTaskIterator> ScanTaskIteratorFromRecordBatch(
    std::vector<std::shared_ptr<RecordBatch>> batches,
    std::shared_ptr<ScanOptions> options, std::shared_ptr<ScanContext> context) {
  ScanTaskVector tasks{std::make_shared<InMemoryScanTask>(batches, std::move(options),
                                                          std::move(context))};
  return MakeVectorIterator(std::move(tasks));
}

ScannerBuilder::ScannerBuilder(std::shared_ptr<Dataset> dataset,
                               std::shared_ptr<ScanContext> scan_context)
    : dataset_(std::move(dataset)),
      fragment_(nullptr),
      scan_options_(std::make_shared<ScanOptions>()),
      scan_context_(std::move(scan_context)) {
  scan_options_->dataset_schema = dataset_->schema();
  DCHECK_OK(Filter(literal(true)));
}

ScannerBuilder::ScannerBuilder(std::shared_ptr<Schema> schema,
                               std::shared_ptr<Fragment> fragment,
                               std::shared_ptr<ScanContext> scan_context)
    : dataset_(nullptr),
      fragment_(std::move(fragment)),
      scan_options_(std::make_shared<ScanOptions>()),
      scan_context_(std::move(scan_context)) {
  scan_options_->dataset_schema = std::move(schema);
  DCHECK_OK(Filter(literal(true)));
}

const std::shared_ptr<Schema>& ScannerBuilder::schema() const {
  return scan_options_->dataset_schema;
}

Status ScannerBuilder::Project(std::vector<std::string> columns) {
  return SetProjection(scan_options_.get(), std::move(columns));
}

Status ScannerBuilder::Project(std::vector<Expression> exprs,
                               std::vector<std::string> names) {
  return SetProjection(scan_options_.get(), std::move(exprs), std::move(names));
}

Status ScannerBuilder::Filter(const Expression& filter) {
  return SetFilter(scan_options_.get(), filter);
}

Status ScannerBuilder::UseThreads(bool use_threads) {
  scan_context_->use_threads = use_threads;
  return Status::OK();
}

Status ScannerBuilder::BatchSize(int64_t batch_size) {
  if (batch_size <= 0) {
    return Status::Invalid("BatchSize must be greater than 0, got ", batch_size);
  }
  scan_options_->batch_size = batch_size;
  return Status::OK();
}

Result<std::shared_ptr<Scanner>> ScannerBuilder::Finish() {
  if (!scan_options_->projection.IsBound()) {
    RETURN_NOT_OK(Project(scan_options_->dataset_schema->field_names()));
  }

  if (dataset_ == nullptr) {
    return std::make_shared<Scanner>(fragment_, scan_options_, scan_context_);
  }
  return std::make_shared<Scanner>(dataset_, scan_options_, scan_context_);
}

using arrow::internal::TaskGroup;

std::shared_ptr<TaskGroup> ScanContext::TaskGroup() const {
  if (use_threads) {
    auto* thread_pool = arrow::internal::GetCpuThreadPool();
    return TaskGroup::MakeThreaded(thread_pool);
  }
  return TaskGroup::MakeSerial();
}

static inline RecordBatchVector FlattenRecordBatchVector(
    std::vector<RecordBatchVector> nested_batches) {
  RecordBatchVector flattened;

  for (auto& task_batches : nested_batches) {
    for (auto& batch : task_batches) {
      flattened.emplace_back(std::move(batch));
    }
  }

  return flattened;
}

struct TableAssemblyState {
  /// Protecting mutating accesses to batches
  std::mutex mutex{};
  std::vector<RecordBatchVector> batches{};

  void Emplace(RecordBatchVector b, size_t position) {
    std::lock_guard<std::mutex> lock(mutex);
    if (batches.size() <= position) {
      batches.resize(position + 1);
    }
    batches[position] = std::move(b);
  }
};

Result<std::shared_ptr<Table>> Scanner::ToTable() {
  ARROW_ASSIGN_OR_RAISE(auto scan_task_it, Scan());
  auto task_group = scan_context_->TaskGroup();

  /// Wraps the state in a shared_ptr to ensure that failing ScanTasks don't
  /// invalidate concurrently running tasks when Finish() early returns
  /// and the mutex/batches fail out of scope.
  auto state = std::make_shared<TableAssemblyState>();

  size_t scan_task_id = 0;
  for (auto maybe_scan_task : scan_task_it) {
    ARROW_ASSIGN_OR_RAISE(auto scan_task, maybe_scan_task);

    auto id = scan_task_id++;
    task_group->Append([state, id, scan_task] {
      ARROW_ASSIGN_OR_RAISE(auto batch_it, scan_task->Execute());
      ARROW_ASSIGN_OR_RAISE(auto local, batch_it.ToVector());
      state->Emplace(std::move(local), id);
      return Status::OK();
    });
  }

  // Wait for all tasks to complete, or the first error.
  RETURN_NOT_OK(task_group->Finish());

  return Table::FromRecordBatches(scan_options_->projected_schema,
                                  FlattenRecordBatchVector(std::move(state->batches)));
}

struct ToBatchesState {
  explicit ToBatchesState(size_t n_tasks)
      : batches(n_tasks), task_drained(n_tasks, false) {}

  /// Protecting mutating accesses to batches
  std::mutex mutex;
  std::condition_variable ready;
  std::vector<std::deque<std::shared_ptr<RecordBatch>>> batches;
  std::vector<bool> task_drained;
  size_t pop_cursor = 0;

  void Push(std::shared_ptr<RecordBatch> b, size_t i_task) {
    std::lock_guard<std::mutex> lock(mutex);
    ready.notify_one();
    if (batches.size() <= i_task) {
      batches.resize(i_task + 1);
      task_drained.resize(i_task + 1);
    }
    batches[i_task].push_back(std::move(b));
  }

  Status Finish(size_t position) {
    std::lock_guard<std::mutex> lock(mutex);
    task_drained[position] = true;
    return Status::OK();
  }

  std::shared_ptr<RecordBatch> Pop() {
    std::unique_lock<std::mutex> lock(mutex);
    ready.wait(lock, [this] {
      while (pop_cursor < batches.size()) {
        // queue for current scan task contains at least one batch, pop that
        if (!batches[pop_cursor].empty()) return true;

        // queue is empty but will be appended to eventually, wait for that
        if (!task_drained[pop_cursor]) return false;

        ++pop_cursor;
      }
      // all scan tasks drained, terminate
      return true;
    });

    if (pop_cursor == batches.size()) return nullptr;

    auto batch = std::move(batches[pop_cursor].front());
    batches[pop_cursor].pop_front();
    return batch;
  }
};

Result<RecordBatchIterator> Scanner::ToBatches() {
  ARROW_ASSIGN_OR_RAISE(auto scan_task_it, Scan());
  ARROW_ASSIGN_OR_RAISE(auto scan_task_vector, scan_task_it.ToVector());

  auto task_group = scan_context_->TaskGroup();
  auto state = std::make_shared<ToBatchesState>(scan_task_vector.size());

  size_t scan_task_id = 0;
  for (auto scan_task : scan_task_vector) {
    auto id = scan_task_id++;
    task_group->Append([state, id, scan_task] {
      ARROW_ASSIGN_OR_RAISE(auto batch_it, scan_task->Execute());
      for (auto maybe_batch : batch_it) {
        ARROW_ASSIGN_OR_RAISE(auto batch, maybe_batch);
        state->Push(std::move(batch), id);
      }
      return state->Finish(id);
    });
  }

  return MakeFunctionIterator(
      [task_group, state]() -> Result<std::shared_ptr<RecordBatch>> {
        if (auto batch = state->Pop()) {
          return batch;
        }
        RETURN_NOT_OK(task_group->Finish());
        return nullptr;
      });
}

Status Scanner::Scan(std::function<Status(std::shared_ptr<RecordBatch>)> visitor) {
  ARROW_ASSIGN_OR_RAISE(auto scan_task_it, Scan());

  auto task_group = scan_context_->TaskGroup();

  for (auto maybe_scan_task : scan_task_it) {
    ARROW_ASSIGN_OR_RAISE(auto scan_task, maybe_scan_task);
    task_group->Append([scan_task, visitor] {
      ARROW_ASSIGN_OR_RAISE(auto batch_it, scan_task->Execute());
      return batch_it.Visit(visitor);
    });
  }

  return task_group->Finish();
}

}  // namespace dataset
}  // namespace arrow
