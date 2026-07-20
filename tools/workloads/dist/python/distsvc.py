# Distributed bank-service — Python asyncio (event loop / epoll), the concurrent svc.py
# SPLIT into two roles over a real network. See ../c/distsvc.c.
#   distsvc.py server            → node B: bind 0.0.0.0:9000, apply transfers, invariant
#   distsvc.py client <peer_ip>  → node A: 6 workers × 500 requests to the server
import asyncio, sys, struct

N, START, CLIENTS, PERCLIENT, PORT = 16, 150, 6, 500, 9000
bal = [START] * N
applied = 0
failed = 0
done_conns = 0

def apply(frm, to, amt):
    global applied, failed
    if bal[frm] >= amt:          # single-threaded event loop: atomic across awaits
        bal[frm] -= amt
        bal[to] += amt
        applied += 1
        return True
    failed += 1
    return False

async def handle(reader, writer):
    global done_conns
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
    done_conns += 1

async def run_server():
    print("CSVC-START python-server", flush=True)
    server = await asyncio.start_server(handle, '0.0.0.0', PORT)
    print("CSVC-LISTEN 9000", flush=True)
    async with server:
        for _ in range(4000):    # ~40s virtual-time cap so a partition can't hang
            if done_conns >= CLIENTS:
                break
            await asyncio.sleep(0.01)
    total = sum(bal)
    membuf = b''.join(struct.pack('<q', bal[i]) for i in range(N))
    h = 0xcbf29ce484222325
    for byte in membuf:
        h = ((h ^ byte) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    print(f"CSVC-APPLIED {applied} CSVC-FAILED {failed}")
    print(f"INV-CONSERVED {str(total == N * START).lower()} (total={total} want=2400)")
    print("CSVC-HASH %016x" % h)
    print("CSVC-EXIT 0", flush=True)

async def client(idx, peer):
    state = (idx + 1) * 1000 + 1
    def nxt():
        nonlocal state
        state = (state * 16807) % 2147483647
        return state
    r = w = None
    for _ in range(80):          # the server may still be booting
        try:
            r, w = await asyncio.open_connection(peer, PORT)
            break
        except OSError:
            await asyncio.sleep(0.1)
    if w is None:
        return
    for _ in range(PERCLIENT):
        frm, to, amt = nxt() % N, nxt() % N, nxt() % 90 + 10
        w.write(f"{frm} {to} {amt}\n".encode())
        await w.drain()
        if not await r.readline():
            break
    w.close()

async def run_client(peer):
    print("CSVC-START python-client", flush=True)
    await asyncio.gather(*[client(c, peer) for c in range(CLIENTS)])
    print("CSVC-CLIENT-DONE", flush=True)

if len(sys.argv) >= 2 and sys.argv[1] == 'server':
    asyncio.run(run_server())
elif len(sys.argv) >= 3 and sys.argv[1] == 'client':
    asyncio.run(run_client(sys.argv[2]))
else:
    print("usage: distsvc.py server | distsvc.py client <peer_ip>")
    sys.exit(1)
