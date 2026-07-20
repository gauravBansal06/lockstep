# Concurrent stateful service — Python asyncio (event loop / epoll). See ../SPEC.md.
import asyncio, os, struct

N, START, CLIENTS, PERCLIENT, SNAPIVL, PORT = 16, 150, 6, 500, 137, 9000
bal = [START] * N
applied = 0
failed = 0
diskfd = None

def snapshot():
    if diskfd is None:
        return
    buf = bytearray(512)
    struct.pack_into('<II', buf, 0, 0x53464244, applied)
    for i in range(N):
        struct.pack_into('<q', buf, 8 + i * 8, bal[i])
    os.pwrite(diskfd, bytes(buf), 0)
    os.fsync(diskfd)

def apply(frm, to, amt):
    global applied, failed
    if bal[frm] >= amt:          # single-threaded event loop: atomic across awaits
        bal[frm] -= amt
        bal[to] += amt
        applied += 1
        if applied % SNAPIVL == 0:
            snapshot()
        return True
    failed += 1
    return False

async def handle(reader, writer):
    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            frm, to, amt = map(int, line.split())
        except ValueError:
            break
        writer.write(b"1\n" if apply(frm, to, amt) else b"0\n")
        await writer.drain()
    writer.close()

async def client(idx):
    state = (idx + 1) * 1000 + 1
    def nxt():
        nonlocal state
        state = (state * 16807) % 2147483647
        return state
    r, w = await asyncio.open_connection('127.0.0.1', PORT)
    for _ in range(PERCLIENT):
        frm, to, amt = nxt() % N, nxt() % N, nxt() % 90 + 10
        w.write(f"{frm} {to} {amt}\n".encode())
        await w.drain()
        if not await r.readline():
            break
    w.close()

async def main():
    global diskfd
    print("CSVC-START python")
    try:
        diskfd = os.open('/dev/vda', os.O_RDWR)
    except OSError:
        diskfd = None
    server = await asyncio.start_server(handle, '127.0.0.1', PORT)
    async with server:
        await asyncio.gather(*[client(c) for c in range(CLIENTS)])
    snapshot()
    total = sum(bal)
    membuf = b''.join(struct.pack('<q', bal[i]) for i in range(N))
    h = 0xcbf29ce484222325
    for byte in membuf:
        h = ((h ^ byte) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    disktotal = -1
    if diskfd is not None:
        rb = os.pread(diskfd, 512, 0)
        if struct.unpack_from('<I', rb, 0)[0] == 0x53464244:
            disktotal = sum(struct.unpack_from('<q', rb, 8 + i * 8)[0] for i in range(N))
    print(f"CSVC-APPLIED {applied} CSVC-FAILED {failed}")
    print(f"INV-CONSERVED {str(total == N * START).lower()} (total={total} want=2400)")
    print(f"INV-DISK-MATCH {str(disktotal == total).lower()} (diskTotal={disktotal})")
    print("CSVC-HASH %016x" % h)
    print("CSVC-EXIT 0")

asyncio.run(main())
