// Distributed bank-service — Node (net / libuv event loop), the concurrent svc.js SPLIT
// into two roles over a real network. See ../c/distsvc.c.
//   distsvc.js server            → node B: bind 0.0.0.0:9000, apply transfers, invariant
//   distsvc.js client <peer_ip>  → node A: 6 workers × 500 requests to the server
const net = require('net');
const N = 16, START = 150, CLIENTS = 6, PERCLIENT = 500, PORT = 9000;
let bal = new Array(N).fill(START);
let applied = 0, failed = 0, doneConns = 0;

function apply(frm, to, amt) {
  if (bal[frm] >= amt) {                 // single-threaded event loop: atomic
    bal[frm] -= amt; bal[to] += amt; applied++;
    return true;
  }
  failed++; return false;
}

function runServer() {
  console.log("CSVC-START node-server");
  const server = net.createServer(sock => {
    sock.setEncoding('utf8');
    let buf = '';
    sock.on('data', d => {
      buf += d;
      let j;
      while ((j = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, j); buf = buf.slice(j + 1);
        const p = line.split(' ');
        sock.write(apply(+p[0], +p[1], +p[2]) ? '1\n' : '0\n');
      }
    });
    sock.on('close', () => { doneConns++; });
    sock.on('error', () => {});
  });
  server.listen(PORT, '0.0.0.0', () => console.log("CSVC-LISTEN 9000"));
  let ticks = 0;
  const timer = setInterval(() => {
    ticks++;
    if (doneConns >= CLIENTS || ticks >= 4000) {   // ~40s virtual-time cap
      clearInterval(timer);
      server.close();
      finish();
    }
  }, 10);
  function finish() {
    const total = bal.reduce((a, b) => a + b, 0);
    const membuf = Buffer.alloc(N * 8);
    for (let i = 0; i < N; i++) membuf.writeBigInt64LE(BigInt(bal[i]), i * 8);
    let h = 0xcbf29ce484222325n;
    const mask = (1n << 64n) - 1n;
    for (const byte of membuf) h = ((h ^ BigInt(byte)) * 0x100000001b3n) & mask;
    console.log(`CSVC-APPLIED ${applied} CSVC-FAILED ${failed}`);
    console.log(`INV-CONSERVED ${total === N * START} (total=${total} want=2400)`);
    console.log("CSVC-HASH " + h.toString(16).padStart(16, '0'));
    console.log("CSVC-EXIT 0");
  }
}

function client(idx, peer) {
  return new Promise(resolve => {
    let state = (idx + 1) * 1000 + 1;
    const nxt = () => { state = (state * 16807) % 2147483647; return state; };
    let sent = 0, recv = 0, rbuf = '', done = false, tries = 0;
    let sock;
    const finish = () => { if (!done) { done = true; if (sock) sock.destroy(); resolve(); } };
    const sendOne = () => {
      if (sent >= PERCLIENT) return;
      sock.write(`${nxt() % N} ${nxt() % N} ${nxt() % 90 + 10}\n`); sent++;
    };
    const connect = () => {
      sock = net.connect(PORT, peer);
      sock.setEncoding('utf8');
      sock.on('connect', sendOne);
      sock.on('data', d => {
        rbuf += d;
        let j;
        while ((j = rbuf.indexOf('\n')) >= 0) {
          rbuf = rbuf.slice(j + 1); recv++;
          if (recv >= PERCLIENT) { finish(); return; }
          sendOne();
        }
      });
      sock.on('error', () => {   // server may still be booting: retry up to 80×
        if (sent === 0 && tries < 80) { tries++; setTimeout(connect, 100); }
        else finish();
      });
      sock.on('close', () => { if (recv < PERCLIENT && sent > 0) finish(); });
    };
    connect();
  });
}

async function runClient(peer) {
  console.log("CSVC-START node-client");
  await Promise.all(Array.from({ length: CLIENTS }, (_, c) => client(c, peer)));
  console.log("CSVC-CLIENT-DONE");
}

const mode = process.argv[2];
if (mode === 'server') runServer();
else if (mode === 'client' && process.argv[3]) runClient(process.argv[3]).catch(e => process.stderr.write("ERR " + e + "\n"));
else { console.log("usage: distsvc.js server | distsvc.js client <peer_ip>"); process.exit(1); }
