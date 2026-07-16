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
import { createHash, randomUUID } from "node:crypto";

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

export type DeadLetterRecord = {
  record: "dead-letter";
  failed_at: string;
  reason: string;
  status?: number;
  event: SpoolEvent;
};

export type RedirectRecord = {
  record: "redirect";
  original_session_id: string;
  recovery_session_id: string;
};

export class DurableSpool {
  private acknowledgements = 0;
  readonly deadLetterPath: string;

  constructor(readonly path: string) {
    this.deadLetterPath = path.endsWith(".jsonl")
      ? `${path.slice(0, -6)}.dead-letter.jsonl`
      : `${path}.dead-letter.jsonl`;
    mkdirSync(dirname(path), { recursive: true });
    if (!existsSync(path)) {
      writeFileSync(path, "", { encoding: "utf8", mode: 0o600 });
      this.fsyncDirectory();
    }
    chmodSync(path, 0o600);
    if (!existsSync(this.deadLetterPath)) {
      writeFileSync(this.deadLetterPath, "", { encoding: "utf8", mode: 0o600 });
      this.fsyncDirectory();
    }
    chmodSync(this.deadLetterPath, 0o600);
  }

  append(
    event: Omit<SpoolEvent, "record" | "event_id"> & { event_id?: string },
  ): string {
    const { event_id: suppliedId, ...payload } = event;
    const eventId = suppliedId?.trim() || randomUUID().replaceAll("-", "");
    this.appendRecord({ record: "event", event_id: eventId, ...payload });
    return eventId;
  }

  acknowledge(eventId: string): void {
    this.appendRecord({ record: "delivered", event_id: eventId });
    this.acknowledgements += 1;
    if (this.acknowledgements >= 256) this.compact();
  }

  recordRedirect(originalSessionId: string, recoverySessionId: string): void {
    if (this.redirectFor(originalSessionId) === recoverySessionId) return;
    this.appendRecord({
      record: "redirect",
      original_session_id: originalSessionId,
      recovery_session_id: recoverySessionId,
    });
  }

  redirectFor(sessionId: string): string | undefined {
    return this.scan().redirects.get(sessionId);
  }

  redirects(): Record<string, string> {
    return Object.fromEntries(this.scan().redirects);
  }

  deadLetter(event: SpoolEvent, reason: string, status?: number): void {
    this.appendTo(this.deadLetterPath, {
      record: "dead-letter",
      failed_at: new Date().toISOString(),
      reason,
      ...(status === undefined ? {} : { status }),
      event,
    });
    this.acknowledge(event.event_id);
  }

  deadLetters(): DeadLetterRecord[] {
    return readFileSync(this.deadLetterPath, "utf8")
      .split("\n")
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line) as DeadLetterRecord);
  }

  pending(): SpoolEvent[] {
    return [...this.scan().events.values()];
  }

  private scan(): {
    events: Map<string, SpoolEvent>;
    redirects: Map<string, string>;
  } {
    const events = new Map<string, SpoolEvent>();
    const redirects = new Map<string, string>();
    const lines = readFileSync(this.path, "utf8").split("\n");
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index].trim();
      if (!line) continue;
      let record: Record<string, unknown>;
      try {
        record = JSON.parse(line) as Record<string, unknown>;
      } catch (error) {
        if (index >= lines.length - 2) break;
        const eventId = `corrupt-${createHash("sha256")
          .update(`${index + 1}:${line}`)
          .digest("hex")
          .slice(0, 24)}`;
        events.set(eventId, {
          record: "event",
          event_id: eventId,
          kind: "corrupt",
          session_id: "",
          external_session_id: "",
          agent: "",
          metadata: { raw_record: line, line_number: index + 1 },
        } as unknown as SpoolEvent);
        continue;
      }
      if (
        record.record === "redirect" &&
        typeof record.original_session_id === "string" &&
        typeof record.recovery_session_id === "string"
      ) {
        redirects.set(record.original_session_id, record.recovery_session_id);
        continue;
      }
      const eventId = typeof record.event_id === "string" ? record.event_id : "";
      if (!eventId) continue;
      if (record.record === "event") events.set(eventId, record as SpoolEvent);
      if (record.record === "delivered") events.delete(eventId);
    }
    return { events, redirects };
  }

  compact(): void {
    const temporary = `${this.path}.${process.pid}.tmp`;
    const scanned = this.scan();
    const records: Array<RedirectRecord | SpoolEvent> = [
      ...[...scanned.redirects].map(([originalSessionId, recoverySessionId]) => ({
        record: "redirect" as const,
        original_session_id: originalSessionId,
        recovery_session_id: recoverySessionId,
      })),
      ...scanned.events.values(),
    ];
    const content = records.map((record) => JSON.stringify(record)).join("\n");
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
    this.appendTo(this.path, record);
  }

  private appendTo(path: string, record: Record<string, unknown>): void {
    const descriptor = openSync(path, "a", 0o600);
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
