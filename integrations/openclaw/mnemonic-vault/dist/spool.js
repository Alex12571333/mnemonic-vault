import { closeSync, chmodSync, existsSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, writeFileSync, writeSync, } from "node:fs";
import { dirname } from "node:path";
import { randomUUID } from "node:crypto";
export class DurableSpool {
    path;
    acknowledgements = 0;
    constructor(path) {
        this.path = path;
        mkdirSync(dirname(path), { recursive: true });
        if (!existsSync(path)) {
            writeFileSync(path, "", { encoding: "utf8", mode: 0o600 });
            this.fsyncDirectory();
        }
        chmodSync(path, 0o600);
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
                throw error;
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
        const descriptor = openSync(this.path, "a", 0o600);
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
