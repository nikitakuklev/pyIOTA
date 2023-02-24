import logging
from pathlib import Path

l = logging.getLogger(__name__)


def parse_lattice(filepath: Path,
                  verbose: bool = False,
                  verbose_vars: bool = False,
                  merge_dipole_edges: bool = True,
                  dipole_merge_method: int = 2,
                  allow_edgeless_dipoles=False,
                  unsafe: bool = False):
    """
    """
    # Argument verification
    if isinstance(filepath, str):
        filepath = Path(filepath)
    elif isinstance(filepath, Path):
        pass
    else:
        raise Exception

    if not filepath.is_file():
        raise IOError(f'File ({filepath}) does not exist or cannot be read')

    # Parse full namelist
    with open(filepath, 'r') as file:
        line = file.readline().strip()
        while line.endswith('&'):
            line = line[:-1] + file.readline().strip()
        line_len = len(line)

    section_type = 0
    buf = ''  # This copy on extension is horrible for performance, but doesn't matter
    is_literal_mode = False
    name = key = value = None
    d = {}
    for pointer in range(line_len):
        char = line[pointer]
        if section_type == 0:
            if char == ':':
                if is_literal_mode:
                    # advance
                    buf += char
                else:
                    # end of value
                    # we should not be in literal mode
                    assert not is_literal_mode
                    # add to data if column in mask
                    name = buf
                    buf = ''
                    l.debug(f'>El name {name}')
                    section_type += 1
            elif char == '"':
                # literal mode toggle
                is_literal_mode = not is_literal_mode
            elif char == ' ':
                if is_literal_mode:
                    buf += char
                else:
                    pass
            else:
                # any other characted gets added to value
                buf += char

        elif section_type == 1 or section_type == 2:
            if char == ':':
                if is_literal_mode:
                    # advance
                    buf += char
                else:
                    # end of value
                    # we should not be in literal mode
                    assert not is_literal_mode
                    # add to data if column in mask
                    name = buf
                    buf = ''
                    l.debug(f'>El name {name}')
                    section_type += 1
            elif char == ',':
                if section_type == 1:
                    #key = buf
                    #value = None
                    d[key] = None
                    key = value = None
                    buf = ''
                elif section_type == 2:
                    d[key] = buf
                    key = value = None
                    buf = ''
                    section_type = 1
                else:
                    raise Exception
            elif char == '=':
                if section_type == 1:
                    key = buf
                    buf = ''
                    section_type = 2
                else:
                    raise Exception
            elif char == '(':
                if section_type == 2:
                    section_type = 3
                else:
                    raise Exception
            elif char == '"':
                # literal mode toggle
                is_literal_mode = not is_literal_mode
            elif char == ' ':
                if is_literal_mode:
                    buf += char
                else:
                    pass
            else:
                # any other characted gets added to value
                buf += char


def _recursive_parse_tuple(pos, line):
    buf = ''
    pointer = pos
    values = []
    is_literal_mode = False
    while True:
        char = line[pointer]
        if char == ',':
            # end of entry
            assert not is_literal_mode
            values += buf
            buf = ''

        elif char == '*':
            # algebra
            mult1 = int(buf)
            pos2, mult2 = _recursive_parse_tuple(pointer, line)
            values += (mult1 * mult2)
            buf = ''
            pos = pos2

        elif char == '"':
            # literal mode toggle
            is_literal_mode = not is_literal_mode

        elif char == ')':
            # break out of list
            return pointer+1, values

        else:
            buf += char