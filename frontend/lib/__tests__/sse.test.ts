import { describe, expect, it } from "vitest";
import { parseSseChunk } from "../sse";

describe("parseSseChunk", () => {
  it("parses complete events and keeps the partial trailing event in rest", () => {
    const buffer =
      'event: status\ndata: {"stage":"retrieving"}\n\n' +
      'event: token\ndata: {"text":"Xin"}\n\n' +
      'event: token\ndata: {"text":"ch';
    const { events, rest } = parseSseChunk(buffer);
    expect(events).toEqual([
      { event: "status", data: '{"stage":"retrieving"}' },
      { event: "token", data: '{"text":"Xin"}' },
    ]);
    expect(rest).toBe('event: token\ndata: {"text":"ch');
  });

  it("tolerates CRLF line endings", () => {
    const buffer = 'event: done\r\ndata: {"ok":true}\r\n\r\n';
    const { events, rest } = parseSseChunk(buffer);
    expect(events).toEqual([{ event: "done", data: '{"ok":true}' }]);
    expect(rest).toBe("");
  });

  it("joins multiple data lines with newline and defaults event to message", () => {
    const buffer = "data: a\ndata: b\n\n";
    const { events } = parseSseChunk(buffer);
    expect(events).toEqual([{ event: "message", data: "a\nb" }]);
  });

  it("ignores comment lines and returns no events for an empty buffer", () => {
    expect(parseSseChunk("")).toEqual({ events: [], rest: "" });
    const { events, rest } = parseSseChunk(": keep-alive\n\nevent: x\ndata: 1\n\n");
    expect(events).toEqual([{ event: "x", data: "1" }]);
    expect(rest).toBe("");
  });
});
