# Concurrent stateful workload — common spec (all languages)

The `statesvc` pattern ported to **every** runtime, so we prove concurrency determinism
across both the **OS-thread+futex** model (C/C++/Go/Java/Rust/C#) **and** the
**event-loop/epoll** model (Node, Python asyncio) — plus GVL/GIL runtimes (Ruby, Perl,
CPython-threads). Each is validated for: deterministic replay per seed, exploration across
seeds, and an invariant that holds across every explored schedule.

## The service (single process, loopback)

- **State:** `N=16` accounts, each `START=150` → conserved total `2400`.
- **Server:** listens on `127.0.0.1:9000`, handles connections **concurrently** (thread /
  goroutine / async task per connection). Per request it applies a transfer **under a lock**
  *iff* `bal[from] >= amt` (sufficiency check ⇒ non-commutative ⇒ order matters).
- **Clients:** `CLIENTS=6` concurrent workers; each opens one connection and sends
  `PERCLIENT=500` requests, reading one reply each.
- **Disk:** every `SNAPIVL=137` applied transfers (and once at the end, under the lock) the
  full state is written to `/dev/vda` (virtio-blk) and at the end read back and compared.

## Wire protocol (TCP, ASCII)

- request line: `"<from> <to> <amt>\n"` (space-separated decimals)
- reply line:   `"1\n"` (applied) or `"0\n"` (insufficient funds)

## PRNG — identical in every language (Park-Miller minstd, JS-safe)

```
state := (client_index + 1) * 1000 + 1          # distinct, non-zero, in [1, 2^31-2]
next() := state = (state * 16807) mod 2147483647 ; return state
per request: from = next() mod N ; to = next() mod N ; amt = next() mod 90 + 10
```
All intermediate products stay < 2^53, so plain doubles (JS) compute it exactly. The op
SET is therefore identical across languages; only the runtime's **interleaving** differs.

## Disk sector 0 (512 bytes, little-endian)

`magic u32 = 0x5346_4244 ("DBFS")` · `applied u32` · `N × i64 balances`.

## Output contract (stdout, exact tokens)

```
CSVC-START <lang>
CSVC-APPLIED <n> CSVC-FAILED <n>
INV-CONSERVED <true|false> (total=<t> want=2400)
INV-DISK-MATCH <true|false> (diskTotal=<t>)
CSVC-HASH <16 hex>          # FNV-1a 64-bit over the N i64 balances, little-endian
CSVC-EXIT 0
```
`CSVC-HASH` = FNV-1a/64 (offset `0xcbf29ce484222325`, prime `0x100000001b3`) over the N
accounts serialized little-endian i64 — identical formula everywhere, so a fixed seed gives
one replayable value per language; different seeds explore different values.

## Harness (initrd)

`insmod virtio_blk.ko` (alpine `virt` kernel: blk is a module), `mount devtmpfs`, bring `lo`
up, then run the program. Run with `--disk <1MB img>` and reset the img to zero per run.
Validate: seed1×2 byte-identical (determinism); seeds 1/2/3 + `SF_SCHED_JITTER` differ
(exploration); `INV-*` true in every run.
