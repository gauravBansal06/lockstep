// Concurrent stateful service — Node (net / libuv event loop). See ../SPEC.md.
const net = require('net');
const fs = require('fs');
const N = 16, START = 150, CLIENTS = 6, PERCLIENT = 500, SNAPIVL = 137, PORT = 9000;
let bal = new Array(N).fill(START);
let applied = 0, failed = 0, diskfd = null;

function snapshot() {
  if (diskfd === null) return;
  const buf = Buffer.alloc(512);
  buf.writeUInt32LE(0x53464244, 0);
  buf.writeUInt32LE(applied, 4);
  for (let i = 0; i < N; i++) buf.writeBigInt64LE(BigInt(bal[i]), 8 + i * 8);
  fs.writeSync(diskfd, buf, 0, 512, 0);
  fs.fsyncSync(diskfd);
}
function apply(frm, to, amt) {
  if (bal[frm] >= amt) {                 // single-threaded event loop: atomic
    bal[frm] -= amt; bal[to] += amt; applied++;
    if (applied % SNAPIVL === 0) snapshot();
    return true;
  }
  failed++; return false;
}

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
  sock.on('error', () => {});
});

function client(idx) {
  return new Promise(resolve => {
    let state = (idx + 1) * 1000 + 1;
    const nxt = () => { state = (state * 16807) % 2147483647; return state; };
    let sent = 0, recv = 0, rbuf = '', done = false;
    const sock = net.connect(PORT, '127.0.0.1');
    sock.setEncoding('utf8');
    const finish = () => { if (!done) { done = true; sock.destroy(); resolve(); } };
    const sendOne = () => {
      if (sent >= PERCLIENT) return;
      sock.write(`${nxt() % N} ${nxt() % N} ${nxt() % 90 + 10}\n`); sent++;
    };
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
    sock.on('error', finish);
    sock.on('close', finish);
  });
}

async function main() {
  console.log("CSVC-START node");
  try { diskfd = fs.openSync('/dev/vda', 'r+'); } catch (e) { diskfd = null; }
  await new Promise(res => server.listen(PORT, '127.0.0.1', res));
  await Promise.all(Array.from({ length: CLIENTS }, (_, c) => client(c)));
  server.close();
  snapshot();
  const total = bal.reduce((a, b) => a + b, 0);
  const membuf = Buffer.alloc(N * 8);
  for (let i = 0; i < N; i++) membuf.writeBigInt64LE(BigInt(bal[i]), i * 8);
  let h = 0xcbf29ce484222325n;
  const mask = (1n << 64n) - 1n;
  for (const byte of membuf) h = ((h ^ BigInt(byte)) * 0x100000001b3n) & mask;
  let disktotal = -1;
  if (diskfd !== null) {
    const rb = Buffer.alloc(512); fs.readSync(diskfd, rb, 0, 512, 0);
    if (rb.readUInt32LE(0) === 0x53464244) {
      disktotal = 0;
      for (let i = 0; i < N; i++) disktotal += Number(rb.readBigInt64LE(8 + i * 8));
    }
  }
  console.log(`CSVC-APPLIED ${applied} CSVC-FAILED ${failed}`);
  console.log(`INV-CONSERVED ${total === N * START} (total=${total} want=2400)`);
  console.log(`INV-DISK-MATCH ${disktotal === total} (diskTotal=${disktotal})`);
  console.log("CSVC-HASH " + h.toString(16).padStart(16, '0'));
  console.log("CSVC-EXIT 0");
}
process.on('unhandledRejection', e => process.stderr.write("UNHANDLED " + (e && e.stack || e) + "\n"));
main().catch(e => process.stderr.write("MAIN-ERR " + (e && e.stack || e) + "\n"));
