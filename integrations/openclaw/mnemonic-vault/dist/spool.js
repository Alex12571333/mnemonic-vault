import { closeSync, chmodSync, existsSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, writeFileSync, writeSync, } from "node:fs";
import { dirname } from "node:path";
import { createHash, randomUUID } from "node:crypto";
export class DurableSpool {
    path;
    acknowledgements = 0;
    deadLetterPath;
    constructor(path) {
        this.path = path;
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
    append(event) {
        const eventId = randomUUID().replaceAll("-", "");
        this.appendRecord({ record: "event", event_id: eventId, ...event });
        return eventId;
    }
    acknowledge(eventId) {
        this.appendRecord({ record: "delivered", event_id: eventId });
        this.acknowledgements += 1;
        if (this.acknowledgements >= 256)
            this.compact();
    }
    deadLetter(event, reason, status) {
        this.appendTo(this.deadLetterPath, {
            record: "dead-letter",
            failed_at: new Date().toISOString(),
            reason,
            ...(status === undefined ? {} : { status }),
            event,
        });
        this.acknowledge(event.event_id);
    }
    deadLetters() {
        return readFileSync(this.deadLetterPath, "utf8")
            .split("\n")
            .filter((line) => line.trim())
            .map((line) => JSON.parse(line));
    }
    pending() {
        const events = new Map();
        const lines = readFileSync(this.path, "utf8").split("\n");
        for (let index = 0; index < lines.length; index += 1) {
            const line = lines[index].trim();
            if (!line)
                continue;
            let record;
            try {
                record = JSON.parse(line);
            }
            catch (error) {
                if (index >= lines.length - 2)
                    break;
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
                });
                continue;
            }
            const eventId = typeof record.event_id === "string" ? record.event_id : "";
            if (!eventId)
                continue;
            if (record.record === "event")
                events.set(eventId, record);
            if (record.record === "delivered")
                events.delete(eventId);
        }
        return [...events.values()];
    }
    compact() {
        const temporary = `${this.path}.${process.pid}.tmp`;
        const content = this.pending().map((event) => JSON.stringify(event)).join("\n");
        writeFileSync(temporary, content ? `${content}\n` : "", { encoding: "utf8", mode: 0o600 });
        const descriptor = openSync(temporary, "r");
        try {
            fsyncSync(descriptor);
        }
        finally {
            closeSync(descriptor);
        }
        renameSync(temporary, this.path);
        this.fsyncDirectory();
        this.acknowledgements = 0;
    }
    appendRecord(record) {
        this.appendTo(this.path, record);
    }
    appendTo(path, record) {
        const descriptor = openSync(path, "a", 0o600);
        try {
            writeSync(descriptor, `${JSON.stringify(record)}\n`, undefined, "utf8");
            fsyncSync(descriptor);
        }
        finally {
            closeSync(descriptor);
        }
    }
    fsyncDirectory() {
        const directory = openSync(dirname(this.path), "r");
        try {
            fsyncSync(directory);
        }
        finally {
            closeSync(directory);
        }
    }
}
