import {
  closeSync,
  chmodSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { dirname } from "node:path";
import { randomUUID } from "node:crypto";

export type SpoolEvent = {
  record: "event";
  event_id: string;
  kind: "message" | "end";
  session_id: string;
  external_session_id: string;
  agent: string;
  role?: string;
  content?: string;
  metadata?: Record<string, unknown>;
};

export class DurableSpool {
  private acknowledgements = 0;

  constructor(readonly path: string) {
    mkdirSync(dirname(path), { recursive: true });
    if (!existsSync(path)) {
      writeFileSync(path, "", { encoding: "utf8", mode: 0o600 });
      this.fsyncDirectory();
    }
    chmodSync(path, 0o600);
  }

  append(event: Omit<SpoolEvent, "record" | "event_id">): string {
    const eventId = randomUUID().replaceAll("-", "");
    this.appendRecord({ record: "event", event_id: eventId, ...event });
    return eventId;
  }

  acknowledge(eventId: string): void {
    this.appendRecord({ record: "delivered", event_id: eventId });
    this.acknowledgements += 1;
    if (this.acknowledgements >= 256) this.compact();
  }

  pending(): SpoolEvent[] {
    const events = new Map<string, SpoolEvent>();
    const lines = readFileSync(this.path, "utf8").split("\n");
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index].trim();
      if (!line) continue;
      let record: Record<string, unknown>;
      try {
        record = JSON.parse(line) as Record<string, unknown>;
      } catch (error) {
        if (index >= lines.length - 2) break;
        throw error;
      }
      const eventId = typeof record.event_id === "string" ? record.event_id : "";
      if (!eventId) continue;
      if (record.record === "event") events.set(eventId, record as SpoolEvent);
      if (record.record === "delivered") events.delete(eventId);
    }
    return [...events.values()];
  }

  compact(): void {
    const temporary = `${this.path}.${process.pid}.tmp`;
    const content = this.pending().map((event) => JSON.stringify(event)).join("\n");
    writeFileSync(temporary, content ? `${content}\n` : "", { encoding: "utf8", mode: 0o600 });
    const descriptor = openSync(temporary, "r");
    try {
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
    renameSync(temporary, this.path);
    this.fsyncDirectory();
    this.acknowledgements = 0;
  }

  private appendRecord(record: Record<string, unknown>): void {
    const descriptor = openSync(this.path, "a", 0o600);
    try {
      writeSync(descriptor, `${JSON.stringify(record)}\n`, undefined, "utf8");
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
  }

  private fsyncDirectory(): void {
    const directory = openSync(dirname(this.path), "r");
    try {
      fsyncSync(directory);
    } finally {
      closeSync(directory);
    }
  }
}
