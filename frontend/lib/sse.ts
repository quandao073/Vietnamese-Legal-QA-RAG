export interface SseEvent {
  event: string;
  data: string;
}

/**
 * Parse as many complete SSE events as possible from `buffer`.
 * Events are separated by a blank line. An incomplete trailing event is
 * returned in `rest` so it can be prepended to the next network chunk.
 */
export function parseSseChunk(buffer: string): { events: SseEvent[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: SseEvent[] = [];

  for (const block of blocks) {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line === "" || line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      else if (field === "data") dataLines.push(value);
    }
    if (dataLines.length === 0) continue;
    events.push({ event, data: dataLines.join("\n") });
  }

  return { events, rest };
}
