r"""
A coarse analysis of the doxygen commands present in arrow-cpp:

$ rg -I ' */// (.*)' -r '$1' cpp/src/ | rg '(^|.* )[@\\](\w+).*' -r '$2' | sort | uniq -c | sort -g -r

- Most use \ but some use @
- Most appear at the start of a line, but for example two instances
  of \param don't (which is an error, I think).

- There are plenty of code fences and markdown-specific constructions
  in the /// already, so trike will need support for MyST.

- Write a script which transforms these, leaving conflict markers or something
  anywhere a replacement couldn't be automatically handled. Getting this down to 10
  minutes of manual refactoring is probably enough automation.

- There are two typos which have non-space before \brief
  $ rg -I ' */// (.*)' -r '$1' cpp/src/ | rg '(.*[^ ])[@\\](\w+).*' | rg brief
  /\brief Return whether the partitionings are equal
  :\brief The catalog of the destination table.

- `\brief` can just be deleted; sphinx doesn't care about that.
  (and we'll pick up the typos if we delete the preceding char too)
- `\param[in] name` and `\param name` can be replaced with :param name:
- `\param[in, out]` and `\param[out]` can be replaced with
  :param name (in, out): and sphinx knows what to do with that, close enough
- `\return` can be replaced with :return:
  - `\result` is equivalent
- `\note .*` can be replaced with .. note::
  - also `\details`
- `\since .*` can be replaced with .. versionadded::
- `\see` can be replaced with .. seealso::, probably with a :cpp:any: role
- `\pre` can be replaced with :precondition: and sphinx shows it along with
  the parameters
  - `\post` and `\invariant` can be replaced with :postcondition: and :invariant:
- `\defgroup name summary...` should be replaced with ///.. group::
  - `\addtogroup name` needs extra handling to give it the priority
  - `\ingroup` should be wrapped with ///.. group:: with priority
  - `\name .*` can probably be handled similarly; it's grouping of members within
    a class so once trike has .. group:: we can make sure it groups in whatever
    namespace
- `\class name` can just be deleted
- `\cond` and `\endcond` can just be deleted
- `\throws E` can be replaced with :throws E:
- `\tparam name` can be replaced with :tparam name:
- `\code`...`\endcode` must be replaced with code fences
- `\deprecated .*` can be replaced with :deprecated:, but will require some processing
  to get the versions first
- I don't know what `\par` is, probably just delete it
- `\internal` can die

What I don't think we can automate is handling all the :undoc-members: - those will need
some explicit doc comments or reviewer permission to drop them out.

docs/source/cpp/api/utilities.rst-.. doxygengroup:: type-traits
docs/source/cpp/api/utilities.rst-.. doxygengroup:: c-type-traits
docs/source/cpp/api/utilities.rst-.. doxygengroup:: type-predicates
docs/source/cpp/api/utilities.rst-.. doxygengroup:: runtime-type-predicates
docs/source/cpp/api/array.rst-.. doxygenclass:: arrow::ArrayVisitor
docs/source/cpp/api/scalar.rst-.. doxygengroup:: concrete-scalar-classes
docs/source/cpp/api/scalar.rst-.. doxygenclass:: arrow::ScalarVisitor
docs/source/cpp/api/datatype.rst-.. doxygenclass:: arrow::FieldPath
docs/source/cpp/api/datatype.rst-.. doxygenclass:: arrow::FieldRef
docs/source/cpp/api/datatype.rst-.. doxygenclass:: arrow::TypeVisitor
docs/source/cpp/api/acero.rst-.. doxygengroup:: acero-api
docs/source/cpp/api/acero.rst-.. doxygengroup:: acero-nodes
docs/source/cpp/api/acero.rst-.. doxygengroup:: acero-internals
docs/source/cpp/api/compute.rst-.. doxygengroup:: compute-concrete-options
docs/source/cpp/api/compute.rst-.. doxygengroup:: expression-convenience
"""

from pathlib import Path
from typing import Iterator

import multiprocessing
import math
import re


def main():
    sources = [
        *Path("cpp/src").glob("**/*.h"),
        *Path("cpp/src").glob("**/*.cc"),
        *Path("cpp/examples").glob("**/*.h"),
        *Path("cpp/examples").glob("**/*.cc"),
        *Path("format").glob("**/*.h"),
        *Path("python/pyarrow/src/arrow").glob("**/*.h"),
        *Path("python/pyarrow/src/arrow").glob("**/*.cc"),
    ]
    with multiprocessing.Pool() as pool:
        chunksize = int(math.sqrt(len(sources)))
        print(f"{chunksize=}")
        for r in pool.imap_unordered(handle_comments, sources, chunksize=chunksize):
            if r:
                print(r)


def handle_comments(path: Path):
    #if "table." not in str(path):
    #if "/bridge.h" not in str(path):
    # if "/hdfs.h" not in str(path):
    #     return ""

    count = 0
    in_lines = path.open()
    out_lines = []
    comment_lines = []
    for line in in_lines:
        count += len(line)
        if not line.lstrip().startswith("///"):
            out_lines.append(line)
            continue

        indent = len(line) - len(line.lstrip())
        comment_lines = [line[indent + 4 :]]
        for line in in_lines:
            if not line.lstrip().startswith("///"):
                break
            comment_lines.append(line[indent + 4 :])
        else:
            line = None

        prefix = f"{' ' * indent}///"
        out_lines.extend(
            f"{prefix} {line}" if line else prefix + "\n"
            for line in handle_comment(comment_lines)
        )
        if line is None:
            break
        out_lines.append(line)

    path.write_text("".join(out_lines))
    return f"{path}: {count}"


DELETE_BRIEF = re.compile(r"[\\@]brief ")
REPLACE_PARAM = re.compile(r"[\\@]param ?(?:|\[in\])(|\[in, ?out\]|\[out\]) (\w+)")
REPLACE_RETURN = re.compile(r"[\\@](returns?|result) ")
REPLACE_THROW = re.compile(r"[\\@]throws? (\w+)")
REPLACE_TPARAM = re.compile(r"[\\@]tparam (\w+)")

REPLACE_PRE = re.compile(r"[\\@]pre ")
REPLACE_POST = re.compile(r"[\\@]post ")
REPLACE_INVARIANT = re.compile(r"[\\@]invariant ")

DELETE_LINES = re.compile(r"[\\@](cond|endcond|class)")


def handle_comment(lines: list[str]) -> Iterator[str]:
    note_pending = False
    for line in lines:
        if line == "":
            if note_pending:
                note_pending = False
                yield "```\n"
                yield ""
                continue

        for note_tag in ["note", "details"]:
            if line.startswith(f"\\{note_tag}"):
                yield "```{note}\n"
                yield line.removeprefix(f"\\{note_tag} ")
                note_pending = True
        if note_pending:
            continue

        if DELETE_LINES.match(line):
            continue

        if line.startswith("\\since"):
            yield "```{versionadded}" + line.removeprefix("\\since")
            yield "```\n"
            continue

        if line.startswith("\\code") or line.startswith("\\endcode"):
            yield "```\n"
            continue

        if line.startswith("\\see"):
            yield "```{seealso}\n"
            yield line.removeprefix("\\see ")
            yield "```\n"
            continue

        line = DELETE_BRIEF.sub("", line)
        line = REPLACE_PARAM.sub(r":param \2\1:", line)
        line = REPLACE_THROW.sub(r":throws \1:", line)
        line = REPLACE_TPARAM.sub(r":param \1:", line)
        line = REPLACE_RETURN.sub(":return: ", line)

        line = REPLACE_PRE.sub(":precondition: ", line)
        line = REPLACE_POST.sub(":postcondition: ", line)
        line = REPLACE_INVARIANT.sub(":invariant: ", line)
        yield line


if __name__ == "__main__":
    main()
