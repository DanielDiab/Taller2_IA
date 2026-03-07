import sys
import time
import tkinter
from typing import Any

_root_window: tkinter.Tk | None = None
_canvas: tkinter.Canvas | None = None
_canvas_xs: int | None = None
_canvas_ys: int | None = None
_canvas_x: int | None = None
_canvas_y: int | None = None


def formatColor(r: float, g: float, b: float) -> str:
    """
    Convert RGB values in [0,1] to a hex color string.
    """
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def sleep(secs: float) -> None:
    """
    Sleep for the given number of seconds, or until a key is pressed.
    """
    global _root_window
    if _root_window is None:
        time.sleep(secs)
    else:
        _root_window.update_idletasks()
        _root_window.after(int(1000 * secs), _root_window.quit)
        _root_window.mainloop()


def _setup_window_bindings() -> None:
    """
    Set up key event bindings for the graphics window.
    """
    assert _root_window is not None
    _root_window.bind("<KeyPress>", _keypress)
    _root_window.bind("<KeyRelease>", _keyrelease)
    _root_window.bind("<FocusIn>", _clear_keys)
    _root_window.bind("<FocusOut>", _clear_keys)
    _clear_keys()


def begin_graphics(
    width: int = 640,
    height: int = 480,
    color: str = formatColor(0, 0, 0),
    title: str | None = None,
) -> None:
    """
    Create a graphics window with the given width, height, background color, and title.
    """
    global \
        _root_window, \
        _canvas, \
        _canvas_x, \
        _canvas_y, \
        _canvas_xs, \
        _canvas_ys, \
        _bg_color

    if _root_window is not None:
        _root_window.destroy()

    _canvas_xs, _canvas_ys = width - 1, height - 1
    _canvas_x, _canvas_y = 0, _canvas_ys
    _bg_color = color

    _root_window = tkinter.Tk()
    _root_window.protocol("WM_DELETE_WINDOW", _destroy_window)
    _root_window.title(title or "Graphics Window")
    _root_window.resizable(False, False)

    try:
        _canvas = tkinter.Canvas(_root_window, width=width, height=height)
        _canvas.pack()
        draw_background()
        _canvas.update()
    except Exception:
        _root_window = None
        raise

    _setup_window_bindings()


def begin_graphics_scrollable(
    viewport_width: int,
    viewport_height: int,
    content_width: int,
    content_height: int,
    color: str = formatColor(0, 0, 0),
    title: str | None = None,
) -> None:
    """
    Create a graphics window with scrollbars when content is larger than viewport.
    Drawing uses content coordinates (0,0) to (content_width-1, content_height-1).
    """
    global \
        _root_window, \
        _canvas, \
        _canvas_x, \
        _canvas_y, \
        _canvas_xs, \
        _canvas_ys, \
        _bg_color

    if _root_window is not None:
        _root_window.destroy()

    _canvas_xs = content_width - 1
    _canvas_ys = content_height - 1
    _canvas_x, _canvas_y = 0, _canvas_ys
    _bg_color = color

    _root_window = tkinter.Tk()
    _root_window.protocol("WM_DELETE_WINDOW", _destroy_window)
    _root_window.title(title or "Graphics Window")
    _root_window.resizable(True, True)

    try:
        frame = tkinter.Frame(_root_window)
        frame.pack(fill=tkinter.BOTH, expand=True)

        vscroll = tkinter.Scrollbar(frame)
        hscroll = tkinter.Scrollbar(frame, orient=tkinter.HORIZONTAL)

        _canvas = tkinter.Canvas(
            frame,
            width=viewport_width,
            height=viewport_height,
            scrollregion=(0, 0, content_width, content_height),
            yscrollcommand=vscroll.set,
            xscrollcommand=hscroll.set,
        )
        vscroll.config(command=_canvas.yview)  # type: ignore[no-untyped-call]
        hscroll.config(command=_canvas.xview)  # type: ignore[no-untyped-call]

        vscroll.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        hscroll.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        _canvas.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)

        draw_background()
        _canvas.update()
    except Exception:
        _root_window = None
        raise

    _setup_window_bindings()


def draw_background() -> None:
    """
    Fill the background with the specified color.
    """
    assert _canvas_ys is not None and _canvas_xs is not None
    corners: list[tuple[int | float, int | float]] = [
        (0, 0),
        (0, _canvas_ys),
        (_canvas_xs, _canvas_ys),
        (_canvas_xs, 0),
    ]
    polygon(corners, _bg_color, fillColor=_bg_color, filled=True, smoothed=False)


def _destroy_window(event: Any = None) -> None:
    """
    Handle window close event by destroying the window and exiting the program.
    """
    sys.exit(0)


def end_graphics() -> None:
    """
    End the graphics session by destroying the window and cleaning up state.
    """
    global _root_window, _canvas, _mouse_enabled
    try:
        try:
            sleep(1)
            if _root_window is not None:
                _root_window.destroy()
        except SystemExit as e:
            print("Ending graphics raised an exception:", e)
    finally:
        _root_window = None
        _canvas = None
        _mouse_enabled = 0
        _clear_keys()


def polygon(
    coords: list[tuple[int | float, int | float]],
    outlineColor: str,
    fillColor: str | None = None,
    filled: bool | int = 1,
    smoothed: bool | int = 1,
    behind: int = 0,
    width: int = 1,
) -> int:
    """
    Draw a polygon with the given vertex coordinates, outline color, fill color, and styling options.
    """
    assert _canvas is not None
    c: list[float] = []
    for coord in coords:
        c.append(coord[0])
        c.append(coord[1])
    if fillColor is None:
        fillColor = outlineColor
    if filled == 0:
        fillColor = ""
    poly = _canvas.create_polygon(
        c, outline=outlineColor, fill=fillColor, smooth=bool(smoothed), width=width
    )
    if behind > 0:
        _canvas.tag_lower(poly, behind)
    return poly


def square(
    pos: tuple[int | float, int | float],
    r: int | float,
    color: str,
    filled: int = 1,
    behind: int = 0,
) -> int:
    """
    Draw a square centered at the given position with the specified size, color, and styling options.
    """
    x, y = pos
    coords = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
    return polygon(coords, color, color, filled, 0, behind=behind)


def circle(
    pos: tuple[int | float, int | float],
    r: int | float,
    outlineColor: str,
    fillColor: str | None = None,
    endpoints: list[int] | None = None,
    style: str = "pieslice",
    width: int = 2,
) -> int:
    """
    Draw a circle (or arc) centered at the given position with the specified radius, colors, and styling options.
    """
    assert _canvas is not None
    x, y = pos
    x0, x1 = x - r - 1, x + r
    y0, y1 = y - r - 1, y + r
    if endpoints is None:
        e = [0, 359]
    else:
        e = list(endpoints)
    while e[0] > e[1]:
        e[1] = e[1] + 360

    return _canvas.create_arc(  # type: ignore[no-untyped-call]
        x0,
        y0,
        x1,
        y1,
        outline=outlineColor,
        fill=fillColor or outlineColor,
        extent=e[1] - e[0],
        start=e[0],
        style=style,
        width=width,
    )


def refresh() -> None:
    """
    Refresh the graphics window to show any updates.
    """
    assert _canvas is not None
    _canvas.update_idletasks()


def edit(id: int, *args: Any) -> None:
    """
    Edit the properties of an existing graphics object by its ID.
    """
    assert _canvas is not None
    _canvas.itemconfigure(id, **dict(args))


def text(
    pos: tuple[int | float, int | float],
    color: str,
    contents: str,
    font: str = "Helvetica",
    size: int = 12,
    style: str = "normal",
    anchor: str = "nw",
) -> int:
    """
    Draw text at the given position with the specified color, font, size, style, and anchor.
    """
    assert _canvas is not None
    x, y = pos
    font_desc = (font, str(size), style)
    return _canvas.create_text(
        x,
        y,
        fill=color,
        text=contents,
        font=font_desc,  # type: ignore[arg-type]
        anchor=anchor,  # type: ignore[arg-type]
    )


def changeText(
    id: int,
    newText: str,
    font: str | None = None,
    size: int = 12,
    style: str = "normal",
) -> None:
    """
    Change the text and optionally font properties of an existing text object by its ID.
    """
    assert _canvas is not None
    _canvas.itemconfigure(id, text=newText)
    if font is not None:
        _canvas.itemconfigure(id, font=(font, "-%d" % size, style))


def line(
    here: tuple[int | float, int | float],
    there: tuple[int | float, int | float],
    color: str = formatColor(0, 0, 0),
    width: int = 2,
) -> int:
    """
    Draw a line from 'here' to 'there' with the specified color and width.
    """
    assert _canvas is not None
    x0, y0 = here[0], here[1]
    x1, y1 = there[0], there[1]
    return _canvas.create_line(x0, y0, x1, y1, fill=color, width=width)


_keysdown: dict[str, int] = {}
_keyswaiting: dict[str, int] = {}
_got_release: int | None = None


def _keypress(event: tkinter.Event) -> None:  # type: ignore[type-arg]
    """
    Handle key press events by updating the keysdown and keyswaiting dictionaries.
    """
    global _got_release

    _keysdown[event.keysym] = 1
    _keyswaiting[event.keysym] = 1
    _got_release = None


def _keyrelease(event: tkinter.Event) -> None:  # type: ignore[type-arg]
    """
    Handle key release events by updating the keysdown and keyswaiting dictionaries.
    """
    global _got_release

    try:
        del _keysdown[event.keysym]
    except Exception:
        pass
    _got_release = 1


def _clear_keys(event: Any = None) -> None:
    """
    Clear all key state by resetting the keysdown and keyswaiting dictionaries.
    """
    global _keysdown, _got_release, _keyswaiting
    _keysdown = {}
    _keyswaiting = {}
    _got_release = None


def keys_pressed() -> Any:
    """
    Return a list of currently pressed keys. This is updated by the key event handlers.
    """
    assert _root_window is not None
    _root_window.update()  # type: ignore[misc]
    if _got_release:
        _root_window.update()  # type: ignore[misc]
    return _keysdown.keys()


def wait_for_keys() -> list[str]:
    """
    Wait until at least one key is pressed and return the list of currently pressed keys.
    """
    keys: list[str] = []
    while keys == []:
        keys = list(keys_pressed())
        sleep(0.05)
    return keys


def remove_from_screen(x: int) -> None:
    """
    Remove a graphics object from the screen by its ID.
    """
    assert _canvas is not None
    _canvas.delete(x)
    if _root_window is not None:
        _root_window.update()  # type: ignore[misc]


def move_to(
    object: int,
    x: float | tuple[float, float],
    y: float | None = None,
) -> None:
    """
    Move a graphics object to the specified (x, y) coordinates. If y is None, x should be a tuple of (x, y).
    """
    assert _canvas is not None
    if y is None:
        try:
            x, y = x  # type: ignore[misc]
        except Exception:
            raise ValueError("incomprehensible coordinates")

    horiz = True
    newCoords: list[float] = []
    current_x, current_y = _canvas.coords(object)[0:2]
    for coord in _canvas.coords(object):
        if horiz:
            inc = x - current_x  # type: ignore[operator]
        else:
            inc = y - current_y  # type: ignore[operator]
        horiz = not horiz

        newCoords.append(coord + inc)  # type: ignore[operator]

    _canvas.coords(object, *newCoords)
    if _root_window is not None:
        _root_window.update()  # type: ignore[misc]
