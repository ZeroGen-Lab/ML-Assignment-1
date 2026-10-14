"""Bundle the allowed student C++ sources into a submission zip.

Public CLI (see docs/student-guide.md):

    python3 -B -m student_tools.package --project starter --output submission.zip

Only ``include/**.hpp|.h``, ``src/**.cpp`` and the exact root documents
``README.md`` and ``AI_USAGE.md`` are packaged. Both documents plus
``include/assignment.hpp`` and ``src/assignment.cpp`` are required.
Documents must contain non-whitespace UTF-8 text. Symlinks (including
root documents, the project root itself, the include/
and src/ base directories and every nested directory), directory
traversal, absolute/backslash paths, NTFS colons / drive letters /
ADS names, Windows reserved device names, trailing dots or spaces in
path components, case-fold and NFC/NFD normalization collisions,
duplicate entries, NUL bytes, non-UTF-8 text and budget violations
are rejected.  An existing output file is never overwritten.

``validate_package(path)`` re-checks a finished archive without executing
anything; later upload services should reuse it as their intake gate.

Limits (also listed in ``--help``): at most MAX_FILES files, MAX_FILE_BYTES
per file, MAX_TOTAL_BYTES uncompressed in total, and a deflate compression
ratio of at most MAX_RATIO.
"""

import argparse
import os
import sys
import unicodedata
import zipfile
from pathlib import Path

MAX_FILES = 64
MAX_FILE_BYTES = 1_048_576            # 1 MiB per file
MAX_TOTAL_BYTES = 8_388_608           # 8 MiB across the package
MAX_RATIO = 200                       # uncompressed / compressed

# Components that collide with Windows device files whatever extension follows.
WINDOWS_RESERVED = frozenset(
    ['CON', 'PRN', 'AUX', 'NUL']
    + [f'COM{i}' for i in range(1, 10)]
    + [f'LPT{i}' for i in range(1, 10)])

DOCUMENTS = ('README.md', 'AI_USAGE.md')
REQUIRED = ('include/assignment.hpp', 'src/assignment.cpp') + DOCUMENTS
ALLOWED_PREFIXES_SUFFIXES = {'include/': ('.hpp', '.h'), 'src/': ('.cpp',)}
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


class PackageError(Exception):
    """A package rule was violated; never carry student content in the message beyond names."""


def _check_component(name, part):
    """Reject path components that are unsafe or unportable on Windows."""
    if ':' in part:
        raise PackageError(f'colon in path component (drive letter or NTFS ADS): {name!r}')
    if part != part.rstrip(' .'):
        raise PackageError(f'trailing dot or space in path component: {name!r}')
    if part.split('.', 1)[0].upper() in WINDOWS_RESERVED:
        raise PackageError(f'Windows reserved device name: {name!r}')


def _check_entry_name(name):
    if not isinstance(name, str) or not name:
        raise PackageError(f'empty entry name: {name!r}')
    if '\\' in name:
        raise PackageError(f'backslash in entry path: {name!r}')
    if name.startswith('/'):
        raise PackageError(f'absolute entry path: {name!r}')
    parts = name.split('/')
    if any(part in ('', '.', '..') for part in parts):
        raise PackageError(f'dot or traversal segment in entry path: {name!r}')
    for part in parts:
        _check_component(name, part)
    if name in DOCUMENTS:
        return name
    prefix = next((p for p in ALLOWED_PREFIXES_SUFFIXES if name.startswith(p)), None)
    if prefix is None:
        raise PackageError(f'entry outside root documents, include/ or src/: {name!r}')
    if not name.endswith(ALLOWED_PREFIXES_SUFFIXES[prefix]):
        raise PackageError(f'entry has an unallowed extension: {name!r}')
    return name


def _check_entry_content(name, size):
    if size > MAX_FILE_BYTES:
        raise PackageError(f'file size {size} exceeds per-file budget of {MAX_FILE_BYTES} '
                           f'bytes: {name}')
    if size == 0:
        raise PackageError(f'empty file: {name}')


def _check_content_bytes(name, data):
    if b'\0' in data:
        raise PackageError(f'NUL byte in text file: {name}')
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        raise PackageError(f'file is not valid UTF-8: {name}') from None
    if name in DOCUMENTS and not text.strip():
        raise PackageError(f'empty document: {name}')


def _check_budget(names, total):
    if len(names) > MAX_FILES:
        raise PackageError(f'file count {len(names)} exceeds limit {MAX_FILES}')
    if total > MAX_TOTAL_BYTES:
        raise PackageError(f'total uncompressed size {total} exceeds limit {MAX_TOTAL_BYTES}')


def _collision_key(name):
    """NFC + casefold: same key for NFC/NFD spellings and case variants."""
    return unicodedata.normalize('NFC', name).casefold()


def _check_case_fold(names):
    seen = {}
    for name in names:
        key = _collision_key(name)
        if key in seen:
            raise PackageError(f'case-insensitive path collision: {seen[key]} and {name}')
        seen[key] = name


def _walk_source_tree(folder, prefix):
    """Explicit scandir walk of one base directory.

    Every entry is lstat-checked: a symlink is rejected outright, never
    followed and never silently skipped.  Only real directories are
    recursed into, so files outside the project tree cannot be collected.
    """
    stack = [(folder, prefix)]
    while stack:
        directory, rel_prefix = stack.pop()
        with os.scandir(directory) as iterator:
            dir_entries = sorted(iterator, key=lambda item: item.name)
        for entry in dir_entries:
            rel = f'{rel_prefix}/{entry.name}'
            if entry.is_symlink():
                raise PackageError(f'symlink rejected: {rel}')
            if entry.is_dir(follow_symlinks=False):
                stack.append((Path(entry.path), rel))
            elif entry.is_file(follow_symlinks=False):
                yield rel, Path(entry.path)


def collect_project(project):
    """Validate the project tree on disk and return sorted (name, bytes) pairs."""
    project = Path(project)
    include, src = project / 'include', project / 'src'
    if project.is_symlink():
        raise PackageError(f'project directory is a symlink, refusing: {project}')
    if not project.is_dir():
        raise PackageError(f'project directory not found: {project}')
    for base_name, base in (('include', include), ('src', src)):
        if base.is_symlink():
            raise PackageError(f'symlink rejected: {base_name}')
        if not base.is_dir():
            raise PackageError(f'project must contain include/ and src/: {project}')

    entries = {}
    for name in DOCUMENTS:
        path = project / name
        if path.is_symlink():
            raise PackageError(f'symlink rejected: {name}')
        if not path.is_file():
            raise PackageError(f'missing required file: {name}')
        data = path.read_bytes()
        _check_entry_content(name, len(data))
        _check_content_bytes(name, data)
        entries[name] = data
    for base_name, base in (('include', include), ('src', src)):
        for rel, path in _walk_source_tree(base, base_name):
            _check_entry_name(rel)
            data = path.read_bytes()
            _check_entry_content(rel, len(data))
            _check_content_bytes(rel, data)
            entries[rel] = data

    names = sorted(entries)
    for required in REQUIRED:
        if required not in entries:
            raise PackageError(f'missing required file: {required}')
    _check_case_fold(names)
    _check_budget(names, sum(len(v) for v in entries.values()))
    return [(name, entries[name]) for name in names]


def validate_package(path):
    """Re-validate a finished archive without executing anything from it."""
    try:
        with zipfile.ZipFile(path) as archive:
            position = archive.fp.tell()
            archive.fp.seek(0, os.SEEK_END)
            archive_size = archive.fp.tell()
            archive.fp.seek(position)
            if archive_size > MAX_TOTAL_BYTES:
                raise PackageError(f'archive size {archive_size} exceeds limit {MAX_TOTAL_BYTES}')
            infos = archive.infolist()
            seen, total = [], 0
            for info in infos:
                name = _check_entry_name(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise PackageError(f'symlink entry rejected: {name}')
                if mode not in (0, 0o100000):
                    raise PackageError(f'non-regular entry rejected: {name}')
                if info.flag_bits & 0x1:
                    raise PackageError(f'encrypted entry rejected: {name}')
                if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise PackageError(f'unsupported compression method rejected: {name}')
                if name in seen:
                    raise PackageError(f'duplicate entry: {name}')
                seen.append(name)
                _check_entry_content(name, info.file_size)
                if info.file_size and info.compress_size:
                    ratio = info.file_size / info.compress_size
                    if ratio > MAX_RATIO:
                        raise PackageError(
                            f'compression ratio {ratio:.0f} exceeds limit {MAX_RATIO}: {name}')
                elif info.file_size and not info.compress_size:
                    raise PackageError(f'zero-size compressed entry with content: {name}')
                total += info.file_size
            for required in REQUIRED:
                if required not in seen:
                    raise PackageError(f'missing required file: {required}')
            _check_case_fold(seen)
            _check_budget(seen, total)
            for info in infos:
                _check_content_bytes(info.filename, archive.read(info))
            return {'files': sorted(seen), 'total_bytes': total}
    except zipfile.BadZipFile as error:
        raise PackageError(f'not a valid zip archive: {error}') from None


def build_package(project, output):
    """Create the submission zip exclusively; validate it before reporting success."""
    entries = collect_project(project)
    output = Path(output)
    try:
        handle = open(output, 'xb')
    except FileExistsError:
        raise PackageError(f'output already exists, use a new file name: {output}') from None
    with handle, zipfile.ZipFile(handle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return validate_package(output)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python3 -m student_tools.package',
        description=('Package README.md, AI_USAGE.md and allowed C++ sources '
                     '(include/*.hpp|*.h, src/*.cpp) into a submission zip. '
                     'Both root documents and include/assignment.hpp, src/assignment.cpp '
                     'are required. Limits: at most %d files, %d bytes per file, %d bytes '
                     'total and per archive, compression ratio at most %d. '
                     'Symlinks, path escapes, case '
                     'collisions, duplicates, NUL bytes and non-UTF-8 text are rejected. '
                     'An existing output file is never overwritten.'
                     % (MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES, MAX_RATIO)))
    parser.add_argument('--project', default='starter',
                        help='project directory containing include/ and src/ (default: starter)')
    parser.add_argument('--output', required=True, help='zip file to create (must not exist)')
    args = parser.parse_args(argv)
    try:
        info = build_package(args.project, args.output)
    except PackageError as error:
        print(f'package: ERROR: {error}', file=sys.stderr)
        return 1
    except OSError as error:
        print(f'package: ERROR: {error}', file=sys.stderr)
        return 1
    for name in info['files']:
        print(f'packaged {name}')
    print(f"package: OK ({len(info['files'])} files, {info['total_bytes']} bytes "
          f"uncompressed) -> {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
