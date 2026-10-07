"""Incremental Server-Sent Events decoder (text in, ``(event, data)`` pairs out)."""
from __future__ import annotations


class SSEDecoder:
    def __init__(self) -> None:
        self._buffer = ""
        self._event: str | None = None
        self._data: list[str] = []

    def feed(self, chunk: str) -> list[tuple[str | None, str]]:
        self._buffer += chunk
        events: list[tuple[str | None, str]] = []
        while True:
            newline = _find_newline(self._buffer)
            if newline is None:
                break
            index, width = newline
            line, self._buffer = self._buffer[:index], self._buffer[index + width:]
            event = self._line(line)
            if event is not None:
                events.append(event)
        return events

    def flush(self) -> list[tuple[str | None, str]]:
        events = self.feed("\n") if self._buffer else []
        event = self._line("")
        if event is not None:
            events.append(event)
        return events

    def _line(self, line: str) -> tuple[str | None, str] | None:
        if line == "":
            if not self._data:
                self._event = None
                return None
            event = (self._event, "\n".join(self._data))
            self._event, self._data = None, []
            return event
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        return None


def _find_newline(text: str) -> tuple[int, int] | None:
    for index, char in enumerate(text):
        if char == "\n":
            return index, 1
        if char == "\r":
            if index + 1 < len(text):
                return index, 2 if text[index + 1] == "\n" else 1
            return None  # wait: may be the first half of \r\n
    return None
