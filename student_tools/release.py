"""Assemble the student-facing release zip from a fixed whitelist.

Course-side tool (students receive the result, they do not run this):

    python3 -B -m student_tools.release --student-readme PATH \
        --public-training-dir TRAINING_SOURCE --public-validation-dir VALIDATION_SOURCE \
        --output ml-assignment-1-student.zip

Whitelist (relative to the repository root, nothing else is ever copied):
    starter/include/**.hpp|.h        public API and student helper headers
    starter/src/**.cpp               student template and helper sources
    starter/tools/driver.cpp         the fixed public driver
    public/index.html                the public landing page (exact file,
                                      never a public/** walk)
    public/entry.html -> index.html  project Pages root entry
    public/leaderboard.html, .js      the fixed, relative-resource leaderboard
    data/public/training/           five reviewed UCI CSVs, manifest and license
                                      (legacy four JSON sources remain readable)
    data/public/                    ten fixed UCI diagnostic members and manifest
                                      (legacy validation/ directory stays unchanged)
    explicit student README -> README.md (never the repository root README)
    starter/README.md                student template instructions
    starter/AI_USAGE.md              student-authored AI declaration template
    student_tools/{__init__,__main__,client,identity,intents,intents_windows,
                   package,release,self_report,selftest,transport}.py
    docs/student-guide.md            the student guide
    docs/assignment-scoring-rules-20261010.md
                                      the public scoring rules (exact file,
                                      never a docs/** walk)

A ``manifest.json`` records the release version and a sha256 for every
file. Teacher material (teacher/, non-public data, private_data/, keys,
models, course tooling beyond the eleven student_tools modules) is excluded
by construction. Public data uses fixed member names, never a directory
walk; copying the whole repository is not possible here. Output
files are created exclusively and never overwritten.

Every component of every whitelist path (and the root itself) is
lstat-checked: a symlink anywhere below the root aborts the release, it
is never followed and never silently skipped.
"""

import argparse
import datetime
import hashlib
import html
import json
import os
import stat
import posixpath
import re
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from student_tools.package import PackageError

RELEASE_VERSION = '0.3.1'
PUBLIC_DATA_VERSION = 'course-uci-20261014-v1'
LEGACY_PUBLIC_DATA_VERSION = 'course-single-20261013-v1'
PUBLIC_FILE_LIMIT = 262144
PUBLIC_CSV_FILE_LIMIT = 2 * 1024 * 1024
PUBLIC_TOTAL_LIMIT = 16 * PUBLIC_FILE_LIMIT
PUBLIC_CSV_MANIFEST_SHA256 = '82b0e95cb9f9bc4d16ec3b7d1af7de779741b3b6935786f1e79bbf68736f3c68'
PUBLIC_CSV_IDENTITIES = {
    PUBLIC_DATA_VERSION: (
        PUBLIC_CSV_MANIFEST_SHA256,
        'e5cc6f36a475724c270f8b3032e8e031b688091318bf521d0e69fea429d0b4e1'),
    'course-uci-20261014-v2': (
        '3083588dab34ea65fe258775c7e9a32ffd74dcae44cc0415f253bc60f7c6f3d2',
        '2d7c97b5526897e28cfeb5764307040212611bb84034e01e3338cb1a73878de0'),
    'course-uci-20261014-v3': (
        '97f69d98c8ef6fe0dd10073c4e2dcb2226b12ebbcef54e292909d2184605b483',
        '05eda33c22b9889779998c6211b18cb347ab820d84b8e881504dd0f970db1408'),
}
PUBLIC_CSV_LICENSE_BYTES = {
    PUBLIC_DATA_VERSION: 1999, 'course-uci-20261014-v2': 1999,
    'course-uci-20261014-v3': 2046,
}
PUBLIC_VALIDATION_IDENTITIES = {
    PUBLIC_DATA_VERSION: 'd4b01f6df571942d89aea5c6f0fb7129c6852440d3a1dc6709b178ab741d92e0',
    'course-uci-20261014-v3': 'cfb8d8998f74ac804ad56fe5d51c59db02ece8c19016c9db73e6c24e7c023365',
}
PUBLIC_CSV_TASKS = {task + '-train.csv': task for task in ('lr', 'svm', 'gbdt', 'kmeans', 'hmm')}
PUBLIC_TRAINING_TASKS = {
    'classification.json': ['lr', 'svm'], 'gbdt.json': ['gbdt'],
    'kmeans.json': ['kmeans'], 'hmm.json': ['hmm'],
}
PUBLIC_VALIDATION_TASKS = {
    task + suffix: task for task in ('lr', 'svm', 'gbdt', 'kmeans', 'hmm')
    for suffix in ('.stdin', '.validation.json')
}
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
STUDENT_TOOL_MODULES = ('__init__.py', '__main__.py', 'client.py', 'identity.py',
                        'intents.py', 'intents_windows.py', 'package.py', 'release.py', 'self_report.py',
                        'selftest.py', 'transport.py')
OPTIONAL_FILES = ()
MAPPED_FILES = (('public/entry.html', 'index.html'),)
DOCUMENT_HTML_FILES = (
    ('README.md', 'README.html'),
    ('docs/student-guide.md', 'docs/student-guide.html'),
    ('docs/assignment-scoring-rules-20261010.md',
     'docs/assignment-scoring-rules-20261010.html'),
    ('starter/README.md', 'starter/README.html'),
    ('starter/AI_USAGE.md', 'starter/AI_USAGE.html'),
)
_DOCUMENT_HTML_FILES = DOCUMENT_HTML_FILES
REQUIRED_FILES = ('docs/assignment-scoring-rules-20261010.md',
                  'docs/student-guide.md', 'public/index.html',
                  'starter/README.md', 'starter/AI_USAGE.md',
                  'starter/tools/driver.cpp')
TREE_RULES = (('starter/include', ('.hpp', '.h')), ('starter/src', ('.cpp',)))
BANNED_NAME_PARTS = ('teacher', 'private_data', '__pycache__', '.pyc', '.git',
                     'reference', 'settlement', 'key', 'models')


def _reject_symlink_components(root, relative):
    """Refuse a whitelist path when any component below the root is a symlink."""
    current = Path(root)
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise PackageError(f'symlink rejected in whitelist path: {relative} '
                               f'(component: {part})')


def _walk_whitelist_tree(root, folder, suffixes):
    """Explicit scandir walk of a whitelist directory.

    Every entry is lstat-checked: symlinks abort the release rather than
    being followed or silently skipped, so only files physically inside
    the tree can be collected.
    """
    base = Path(root) / folder
    found = []
    stack = [(base, folder)]
    while stack:
        directory, prefix = stack.pop()
        with os.scandir(directory) as iterator:
            dir_entries = sorted(iterator, key=lambda item: item.name)
        for entry in dir_entries:
            name = f'{prefix}/{entry.name}'
            if entry.is_symlink():
                raise PackageError(f'symlink rejected: {name}')
            if entry.is_dir(follow_symlinks=False):
                stack.append((Path(entry.path), name))
            elif entry.is_file(follow_symlinks=False) and Path(name).suffix in suffixes:
                found.append((name, Path(entry.path)))
    return found


def _validate_document(path, name):
    try:
        text = path.read_bytes().decode('utf-8')
    except UnicodeDecodeError:
        raise PackageError(f'public document must be UTF-8: {name}') from None
    if not text.strip() or '\x00' in text:
        raise PackageError(f'public document must be nonempty text without NUL: {name}')


def _collect(root, student_readme=None):
    root = Path(root)
    if root.is_symlink():
        raise PackageError(f'root is a symlink, refusing: {root}')
    if not root.is_dir():
        raise PackageError(f'root directory not found: {root}')
    names = []
    included = set()

    def take(name, path):
        if not path.is_file() or path.is_symlink():
            raise PackageError(f'whitelist entry missing or not a regular file: {name}')
        if any(part in name for part in BANNED_NAME_PARTS):
            raise PackageError(f'whitelist entry touches banned material: {name}')
        if name in included:
            raise PackageError(f'duplicate release destination: {name}')
        if name.endswith(('.md', '.html')):
            _validate_document(path, name)
        included.add(name)
        names.append((name, path))

    if student_readme is None:
        raise PackageError('explicit student_readme is required; repository README is not public')
    readme = Path(os.path.abspath(student_readme))
    _reject_symlink_components(Path(readme.anchor), readme.relative_to(readme.anchor))
    if readme.resolve() == (root / 'README.md').resolve():
        raise PackageError('repository root README is not student-facing')
    take('README.md', readme)
    for source, destination in MAPPED_FILES:
        _reject_symlink_components(root, source)
        take(destination, root / source)

    for folder, suffixes in TREE_RULES:
        _reject_symlink_components(root, folder)
        base = root / folder
        if not base.is_dir():
            raise PackageError(f'whitelist directory missing: {folder}')
        for name, path in _walk_whitelist_tree(root, folder, suffixes):
            take(name, path)
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        _reject_symlink_components(root, name)
        path = root / name
        if path.is_file():
            take(name, path)
        elif name in REQUIRED_FILES:
            raise PackageError(f'whitelist entry missing: {name}')
    if ((root / 'docs/dataset-columns.md').exists() or
            (root / 'docs/dataset-columns.md').is_symlink()):
        _reject_symlink_components(root, 'docs/dataset-columns.md')
        take('docs/dataset-columns.md', root / 'docs/dataset-columns.md')
    _reject_symlink_components(root, 'student_tools')
    tools = root / 'student_tools'
    for module in STUDENT_TOOL_MODULES:
        _reject_symlink_components(root, f'student_tools/{module}')
        take(f'student_tools/{module}', tools / module)
    return names


def _approved_https_links(links):
    if isinstance(links, (str, bytes)):
        raise PackageError('approved HTTPS links must be a collection of complete URLs')
    result = set()
    for href in links:
        if not isinstance(href, str):
            raise PackageError('approved HTTPS link must be a string')
        try:
            parts = urlsplit(href)
            decoded = unquote(href)
            authority = re.fullmatch(
                r'[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*(?::([1-9][0-9]{0,4}))?',
                parts.netloc)
            valid = (href.startswith('https://') and parts.scheme == 'https'
                     and authority is not None and parts.hostname is not None
                     and (parts.port is None or 1 <= parts.port <= 65535)
                     and not re.search(r'%(?![0-9a-fA-F]{2})', href)
                     and all(33 <= ord(char) <= 126 for char in href)
                     and not any(ord(char) < 32 or ord(char) == 127 for char in decoded)
                     and not any(char in href + decoded for char in '\\<>"'))
        except ValueError:
            valid = False
        if not valid:
            raise PackageError(f'unsafe approved HTTPS link: {href}')
        result.add(href)
    return frozenset(result)


def _document_link(href, name, names, *, rewrite=True, approved_https_links=()):
    if (not href or '\\' in href
            or any(ord(char) < 32 or ord(char) == 127 for char in href)):
        raise PackageError(f'unsafe document link in {name}: {href}')
    try:
        parts = urlsplit(href)
    except ValueError:
        raise PackageError(f'unsafe document link in {name}: {href}') from None
    if parts.scheme or parts.netloc:
        if href in approved_https_links:
            _approved_https_links((href,))
            return href
        raise PackageError(f'unapproved external document link in {name}: {href}')
    decoded = unquote(parts.path)
    if (not href or parts.scheme or parts.netloc or parts.query
            or decoded.startswith('/') or '\\' in decoded
            or any(ord(char) < 32 for char in href + decoded)):
        raise PackageError(f'unsafe document link in {name}: {href}')
    target = posixpath.normpath(posixpath.join(posixpath.dirname(name), decoded)) if decoded else name
    if target not in names:
        raise PackageError(f'unknown document link in {name}: {href}')
    mapped = dict(_DOCUMENT_HTML_FILES).get(target, target) if rewrite else target
    path = posixpath.relpath(mapped, posixpath.dirname(name) or '.') if decoded else ''
    return urlunsplit(('', '', path, '', parts.fragment))


def _inline_document(text, name, names, approved_https_links=()):
    result = []
    while text:
        if text.startswith('`') and '`' in text[1:]:
            end = text.index('`', 1)
            result.append('<code>' + html.escape(text[1:end]) + '</code>')
            text = text[end + 1:]
            continue
        if text.startswith('**') and '**' in text[2:]:
            end = text.index('**', 2)
            result.append('<strong>' + _inline_document(text[2:end], name, names, approved_https_links) + '</strong>')
            text = text[end + 2:]
            continue
        match = re.match(r'\[([^\]]+)\]\(([^\s)]+)\)', text)
        if match:
            label, href = match.groups()
            result.append('<a href="' + html.escape(_document_link(href, name, names, approved_https_links=approved_https_links), quote=True)
                          + '">' + _inline_document(label, name, names, approved_https_links) + '</a>')
            text = text[match.end():]
            continue
        result.append(html.escape(text[0]))
        text = text[1:]
    return ''.join(result)


def _document_blocks(lines, name, names, approved_https_links=()):
    blocks = []
    inline = lambda value: _inline_document(value, name, names, approved_https_links)
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        fence = re.match(r'^```([\w+-]*)\s*$', line)
        if fence:
            code = []
            i += 1
            while i < len(lines) and lines[i].strip() != '```':
                code.append(lines[i])
                i += 1
            if i == len(lines):
                raise PackageError(f'unclosed code fence in {name}')
            language = (' class="language-' + fence.group(1) + '"') if fence.group(1) else ''
            blocks.append('<pre tabindex="0"><code' + language + '>'
                          + html.escape('\n'.join(code)) + '</code></pre>')
            i += 1
            continue
        heading = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2)
            blocks.append(f'<h{level}>' + inline(title) + f'</h{level}>')
            i += 1
            continue
        if (i + 1 < len(lines) and line.strip().startswith('|')
                and re.fullmatch(r'\s*\|\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*', lines[i + 1])):
            cells = lambda row: row.strip().strip('|').split('|')
            header = cells(line)
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = cells(lines[i])
                if len(row) != len(header):
                    raise PackageError(f'inconsistent table in {name}')
                rows.append('<tr>' + ''.join('<td>' + inline(cell.strip()) + '</td>' for cell in row) + '</tr>')
                i += 1
            blocks.append('<div class="table-scroll" role="region" aria-label="文档表格" tabindex="0">'
                          '<table><thead><tr>' + ''.join('<th>' + inline(cell.strip()) + '</th>' for cell in header)
                          + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
            continue
        bullet = re.match(r'^( *)([-*]|\d+\.)\s+(.+)$', line)
        if bullet:
            indent = len(bullet.group(1))
            ordered = bullet.group(2)[0].isdigit()
            tag = 'ol' if ordered else 'ul'
            start = (' start="' + bullet.group(2)[:-1] + '"') if ordered else ''
            items = []
            while i < len(lines):
                current = re.match(r'^( *)([-*]|\d+\.)\s+(.+)$', lines[i])
                if not current or len(current.group(1)) != indent or current.group(2)[0].isdigit() != ordered:
                    break
                content = inline(current.group(3))
                i += 1
                nested = []
                while i < len(lines) and lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) > indent:
                    nested.append(lines[i][indent + 2:])
                    i += 1
                if nested:
                    content += _document_blocks(nested, name, names, approved_https_links)
                items.append('<li>' + content + '</li>')
            blocks.append('<' + tag + start + '>' + ''.join(items) + '</' + tag + '>')
            continue
        paragraph = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#{1,6}\s|```|\s*([-*]|\d+\.)\s)', lines[i]):
            if lines[i].strip().startswith('|'):
                break
            paragraph.append(lines[i].strip())
            i += 1
        blocks.append('<p>' + inline(' '.join(paragraph)) + '</p>')
    return '\n'.join(blocks)


def _render_document(text, source, names, approved_https_links=()):
    mapping = dict(_DOCUMENT_HTML_FILES)
    if source not in mapping:
        raise PackageError(f'non-whitelist document source: {source}')
    name = mapping[source]
    lines = text.splitlines()
    title = next((line[2:] for line in lines if line.startswith('# ')), source)
    labels = ('学生 README', '学生指南', '完整评分规则', '模板说明', 'AI 使用说明')
    navigation = [('public/index.html', '课程说明'), *zip(mapping.values(), labels), (source, 'Markdown 原文')]
    links = ' · '.join('<a href="' + html.escape(_document_link(
        posixpath.relpath(target, posixpath.dirname(name) or '.'), name, names, rewrite=False), quote=True)
                       + '">' + label + '</a>' for target, label in navigation)
    return ('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>'
            + html.escape(title) + '</title>\n<style>\n'
            'body { margin: 0; padding: 24px 16px; font-family: sans-serif; line-height: 1.75; color: #24303c; background: #f6f9fb; }\n'
            'main { max-width: 860px; margin: auto; }\n'
            'p, li, td, th, nav { overflow-wrap: anywhere; }\n'
            'a { color: #1f6f8b; }\n'
            'a:focus-visible, pre:focus-visible, .table-scroll:focus-visible { outline: 3px solid #1f6f8b; outline-offset: 3px; }\n'
            'pre, .table-scroll { max-width: 100%; overflow-x: auto; }\n'
            'pre { padding: 12px; box-sizing: border-box; background: #eaf0f4; }\n'
            'code { font-family: monospace; }\n'
            'table { border-collapse: collapse; width: 100%; }\n'
            'th, td { padding: 8px; border: 1px solid #ccd7df; text-align: left; }\n'
            '@media (max-width: 560px) { body { padding: 16px 12px; } h1 { font-size: 1.55em; } }\n'
            '</style>\n</head>\n<body>\n<main>\n<nav aria-label="文档阅读导航">'
            + links + '</nav>\n' + _document_blocks(lines, name, names, approved_https_links)
            + '\n<nav aria-label="返回文档导航">' + links + '</nav>\n</main>\n</body>\n</html>\n').encode('utf-8')


class _DocumentPageLinks(HTMLParser):
    def __init__(self, text, name, names, approved_https_links=()):
        super().__init__(convert_charrefs=False)
        self.name, self.names = name, names
        self.approved_https_links = approved_https_links
        self.edits = []
        self.line_offsets = [0]
        self.line_offsets.extend(match.end() for match in re.finditer('\n', text))

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == 'href':
                if value is None:
                    raise PackageError(f'missing document link in {self.name}')
                _document_link(value, self.name, self.names, approved_https_links=self.approved_https_links)
        raw = self.get_starttag_text()
        line, column = self.getpos()
        offset = self.line_offsets[line - 1] + column
        attributes = re.compile(r'''\s+([^\s/>=]+)(?:\s*=\s*("[^"]*"|'[^']*'|[^\s>]+))?''')
        for match in attributes.finditer(raw):
            if match.group(1).lower() != 'href':
                continue
            value = match.group(2)
            if value is None:
                raise PackageError(f'missing document link in {self.name}')
            if value[:1] in ('"', "'"):
                value = value[1:-1]
            mapped = _document_link(html.unescape(value), self.name, self.names,
                                    approved_https_links=self.approved_https_links)
            start, end = match.span(2)
            self.edits.append((offset + start, offset + end,
                               '"' + html.escape(mapped, quote=True) + '"'))


def _public_page(text, name, names, approved_https_links=()):
    parser = _DocumentPageLinks(text, name, names, approved_https_links)
    parser.feed(text)
    parser.close()
    for start, end, value in reversed(parser.edits):
        text = text[:start] + value + text[end:]
    return text.encode('utf-8')


def _release_content(entries, approved_https_links=(), public_content=None):
    approved_https_links = _approved_https_links(approved_https_links)
    content = {name: path.read_bytes() for name, path in entries}
    if public_content:
        if set(content) & set(public_content):
            raise PackageError('duplicate public destination')
        content.update(public_content)
    destinations = set(content)
    for source, destination in DOCUMENT_HTML_FILES:
        if (source, destination) not in _DOCUMENT_HTML_FILES:
            raise PackageError(f'non-whitelist document mapping: {source}')
        if source not in content or destination in destinations:
            raise PackageError(f'missing source or duplicate generated destination: {destination}')
        destinations.add(destination)
    for source, destination in DOCUMENT_HTML_FILES:
        content[destination] = _render_document(content[source].decode('utf-8'), source, destinations, approved_https_links)
    for name in ('index.html', 'public/index.html'):
        content[name] = _public_page(content[name].decode('utf-8'), name, destinations, approved_https_links)
    return content


def _public_bytes(path, limit=PUBLIC_FILE_LIMIT):
    path = Path(os.path.abspath(path))
    _reject_symlink_components(Path(path.anchor), path.relative_to(path.anchor))
    try:
        with path.open('rb') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
                raise PackageError(f'invalid public file or budget: {path.name}')
            data = handle.read(limit + 1)
    except OSError:
        raise PackageError(f'public file missing or unreadable: {path.name}') from None
    if not 0 < len(data) <= limit:
        raise PackageError(f'public file exceeds budget: {path.name}')
    return data


def _public_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result

    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeError, RecursionError):
        raise PackageError('invalid public JSON') from None
    if not isinstance(value, dict):
        raise PackageError('public JSON must be an object')
    return value


def _public_csv_source(folder, manifest_bytes, manifest):
    identity = PUBLIC_CSV_IDENTITIES.get(manifest.get('data_version'))
    if identity is None or hashlib.sha256(manifest_bytes).hexdigest() != identity[0]:
        raise PackageError('unreviewed public CSV manifest')
    if (manifest.get('schema') != 'course-uci-training-csv-v1' or
            set(manifest.get('files', {})) != set(PUBLIC_CSV_TASKS)):
        raise PackageError('invalid public CSV manifest')
    content = {'manifest.json': manifest_bytes}
    for name, item in manifest['files'].items():
        if (set(item) != {'bytes', 'sha256'} or type(item['bytes']) is not int or
                not 0 < item['bytes'] <= PUBLIC_CSV_FILE_LIMIT):
            raise PackageError('invalid public CSV member')
        data = _public_bytes(folder / name, PUBLIC_CSV_FILE_LIMIT)
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise PackageError(f'public member identity mismatch: {name}')
        content[name] = data
    license_data = _public_bytes(folder / 'DATA-SOURCES.md')
    if (len(license_data) != PUBLIC_CSV_LICENSE_BYTES[manifest['data_version']] or
            hashlib.sha256(license_data).hexdigest() != identity[1]):
        raise PackageError('public source license identity mismatch')
    content['DATA-SOURCES.md'] = license_data
    return manifest, content


def _public_source(folder, training):
    if folder is None:
        raise PackageError('explicit public_training_dir and public_validation_dir are required')
    folder = Path(os.path.abspath(folder))
    _reject_symlink_components(Path(folder.anchor), folder.relative_to(folder.anchor))
    manifest_bytes = _public_bytes(folder / 'manifest.json')
    manifest = _public_json(manifest_bytes)
    if training and manifest.get('data_version') in PUBLIC_CSV_IDENTITIES:
        return _public_csv_source(folder, manifest_bytes, manifest)
    uci_validation = (not training and
                      manifest.get('data_version') in PUBLIC_VALIDATION_IDENTITIES)
    if (uci_validation and hashlib.sha256(manifest_bytes).hexdigest() !=
            PUBLIC_VALIDATION_IDENTITIES[manifest['data_version']]):
        raise PackageError('unreviewed public UCI validation manifest')
    keys = ({'schema_version', 'data_version', 'dataset_id', 'status', 'files'} if training else
            ({'schema', 'data_version', 'dataset_id', 'files'} if uci_validation else
             {'status', 'data_version', 'dataset_id', 'files', 'input_manifest',
              'input_manifest_sha256', 'driver_executed', 'student_executed', 'gates'}))
    version = manifest.get('data_version')
    versions = ((LEGACY_PUBLIC_DATA_VERSION,) if training else
                (LEGACY_PUBLIC_DATA_VERSION, *PUBLIC_VALIDATION_IDENTITIES))
    expected = PUBLIC_TRAINING_TASKS if training else PUBLIC_VALIDATION_TASKS
    expected_status = ('PUBLIC_TRAINING_CANDIDATE' if training else
                       'PUBLIC_VALIDATION_CANDIDATE_NOT_PUBLISHED')
    if (set(manifest) != keys or
            (manifest.get('schema') != 'course-public-validation-v1' if uci_validation else
             manifest['status'] != expected_status) or
            version not in versions or
            not isinstance(manifest['dataset_id'], str) or
            re.fullmatch(r'[0-9a-f]{64}', manifest['dataset_id']) is None or
            (training and manifest['schema_version'] != 'public-single-training-v1') or
            not isinstance(manifest['files'], list) or len(manifest['files']) != len(expected)):
        raise PackageError('invalid public source manifest')
    content = {}
    item_keys = ({'name', 'tasks', 'bytes', 'sha256', 'records',
                  'local_train_count', 'local_validation_count'} if training else
                 {'name', 'task', 'bytes', 'sha256'})
    for item in manifest['files']:
        if not isinstance(item, dict) or set(item) != item_keys:
            raise PackageError('invalid public manifest member')
        name = item['name']
        if (not isinstance(name, str) or name not in expected or name in content or
                item['tasks' if training else 'task'] != expected[name] or
                type(item['bytes']) is not int or not 0 < item['bytes'] <= PUBLIC_FILE_LIMIT or
                not isinstance(item['sha256'], str) or
                re.fullmatch(r'[0-9a-f]{64}', item['sha256']) is None):
            raise PackageError('invalid public member name, task, size or digest')
        data = _public_bytes(folder / name)
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise PackageError(f'public member identity mismatch: {name}')
        if name.endswith('.json'):
            value = _public_json(data)
            member_keys = ({'schema_version', 'data_version', 'dataset_id', 'tasks',
                            'usage', 'local_split', 'records'} if training else
                           {'data_version', 'dataset_id', 'task', 'evaluation_mode',
                            'validation_ids', 'validation_answers'})
            if (set(value) != member_keys or value['data_version'] != version or
                    value['dataset_id'] != manifest['dataset_id'] or
                    value['tasks' if training else 'task'] != expected[name] or
                    (training and value['schema_version'] != 'public-single-training-v1')):
                raise PackageError(f'invalid public member metadata: {name}')
        content[name] = data
    return manifest, content


class _LeaderboardAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = 0
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if len(values) != len(attrs) or any(key.startswith('on') for key in values):
            raise PackageError('unexpected leaderboard attributes')
        if tag in ('link', 'base', 'iframe', 'object', 'embed', 'img', 'audio', 'video', 'source'):
            raise PackageError('unexpected leaderboard asset')
        if tag == 'script':
            if values != {'src': 'leaderboard.js', 'defer': None}:
                raise PackageError('leaderboard script must be relative and deferred')
            self.scripts += 1
            self.in_script = True
        elif any(key in values for key in ('src', 'srcset', 'href', 'action', 'formaction')):
            raise PackageError('unexpected leaderboard resource link')

    def handle_startendtag(self, tag, attrs):
        if tag == 'script':
            raise PackageError('self-closing leaderboard script rejected')
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_script = False

    def handle_data(self, data):
        if self.in_script and data.strip():
            raise PackageError('inline leaderboard script rejected')


def _public_delivery(root, public_training_dir, public_validation_dir, leaderboard_dir=None):
    training_manifest, training = _public_source(public_training_dir, True)
    validation_manifest, validation = _public_source(public_validation_dir, False)
    if (training_manifest['dataset_id'] != validation_manifest['dataset_id'] or
            training_manifest['data_version'] != validation_manifest['data_version']):
        raise PackageError('public training/validation dataset mismatch')
    public_manifest = {
        'schema': 'course-public-validation-v1', 'data_version': validation_manifest['data_version'],
        'dataset_id': validation_manifest['dataset_id'], 'files': validation_manifest['files'],
    }
    content = {'data/public/training/' + name: data for name, data in training.items()}
    validation_prefix = ('data/public/' if training_manifest['data_version'] in
                         PUBLIC_VALIDATION_IDENTITIES else 'data/public/validation/')
    content.update({validation_prefix + name: data for name, data in validation.items()})
    content[validation_prefix + 'manifest.json'] = (json.dumps(public_manifest, indent=2) + '\n').encode()
    new_version = training_manifest['data_version'] in PUBLIC_VALIDATION_IDENTITIES
    if new_version and leaderboard_dir is None:
        raise PackageError('explicit reviewed leaderboard_dir is required for UCI release')
    asset_root = Path(leaderboard_dir) if leaderboard_dir is not None else Path(root) / 'public'
    identities = {
        'leaderboard.html': (8579, '5805cab11c729802701b73388a018006e740863fda3189abbc3064fdc294dddf'),
        'leaderboard.js': (148452, 'c4d627fded41918fac6b9c84248b823889974c548f0195b042e76533c1c52a15'),
    }
    for name, identity in identities.items():
        data = _public_bytes(asset_root / name)
        if new_version and (len(data), hashlib.sha256(data).hexdigest()) != identity:
            raise PackageError(f'unreviewed UCI leaderboard asset: {name}')
        content['public/' + name] = data
    try:
        page = content['public/leaderboard.html'].decode('utf-8')
        script = content['public/leaderboard.js'].decode('utf-8')
    except UnicodeError:
        raise PackageError('leaderboard assets must be UTF-8') from None
    if not page.strip() or not script.strip() or '\x00' in page + script:
        raise PackageError('invalid leaderboard text')
    assets = _LeaderboardAssets()
    assets.feed(page)
    assets.close()
    if assets.scripts != 1 or assets.in_script or re.search(r'url\s*\(|@import', page, re.I):
        raise PackageError('leaderboard asset set is not closed')
    if sum(map(len, content.values())) > PUBLIC_TOTAL_LIMIT:
        raise PackageError('public delivery exceeds total budget')
    return content


def build_release(root, output, *, student_readme=None, approved_https_links=(),
                  public_training_dir=None, public_validation_dir=None, leaderboard_dir=None):
    """Return (manifest, warnings) from explicit reviewed sources and fixed members."""
    output = Path(output)
    approved_https_links = _approved_https_links(approved_https_links)
    public = _public_delivery(root, public_training_dir, public_validation_dir, leaderboard_dir)
    content = _release_content(_collect(root, student_readme), approved_https_links, public)
    try:
        handle = open(output, 'xb')
    except FileExistsError:
        raise PackageError(f'output already exists, use a new file name: {output}') from None
    manifest = {'release_version': RELEASE_VERSION,
                'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'policy': 'fixed whitelist; teacher and private material excluded',
                'approved_https_links': sorted(approved_https_links),
                'files': []}
    with handle, zipfile.ZipFile(handle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(content.items()):
            info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
            manifest['files'].append({'name': name, 'bytes': len(data),
                                      'sha256': hashlib.sha256(data).hexdigest()})
        info = zipfile.ZipInfo('manifest.json', date_time=ZIP_TIMESTAMP)
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        import json
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    included = set(content)
    missing_optional = [name for name in OPTIONAL_FILES if name not in included]
    return manifest, missing_optional


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python3 -m student_tools.release',
        description=('Assemble the student release zip from the fixed whitelist '
                     '(starter sources, public driver, student guide, scoring '
                     'rules, student_tools). '
                     'Copies only fixed public data; never teacher/ or private material. '
                     'Output is created exclusively and never overwritten.'))
    parser.add_argument('--root', default='.', help='repository root (default: current directory)')
    parser.add_argument('--student-readme', required=True,
                        help=('reviewed student README source, relative to current directory; '
                              'use physical path without symlink components; '
                              'repository root README is rejected'))
    parser.add_argument('--public-training-dir', required=True,
                        help='reviewed source directory containing the five fixed UCI training CSVs, manifest and license')
    parser.add_argument('--public-validation-dir', required=True,
                        help='reviewed source directory containing the ten fixed validation members and manifest')
    parser.add_argument('--leaderboard-dir',
                        help='reviewed D60 static source directory; required for UCI data release')
    parser.add_argument('--output', required=True, help='zip file to create (must not exist)')
    parser.add_argument('--approved-https-link', action='append', default=[],
                        help='reviewed complete HTTPS URL; repeat for each approved URL (default: none)')
    args = parser.parse_args(argv)
    try:
        manifest, missing_optional = build_release(
            args.root, args.output, student_readme=args.student_readme,
            approved_https_links=args.approved_https_link,
            public_training_dir=args.public_training_dir,
            public_validation_dir=args.public_validation_dir,
            leaderboard_dir=args.leaderboard_dir)
    except (PackageError, OSError) as error:
        print(f'release: ERROR: {error}', file=sys.stderr)
        return 1
    print(f"release: {args.output} ({len(manifest['files'])} files, "
          f"version {RELEASE_VERSION})")
    for name in missing_optional:
        print(f'release: WARNING optional whitelist entry absent: {name}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
