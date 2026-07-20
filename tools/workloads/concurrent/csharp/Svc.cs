// Concurrent stateful service — C# (.NET, TcpListener + threads + lock). See ../SPEC.md.
using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

class Svc {
    const int N = 16, START = 150, CLIENTS = 6, PERCLIENT = 500, SNAPIVL = 137, PORT = 9000;
    static long[] bal = new long[N];
    static int applied = 0, failed = 0;
    static readonly object mu = new object();

    // Raw libc: .NET's FileStream/RandomAccess don't round-trip a raw block device
    // (/dev/vda) here, so go straight to pwrite/pread via P/Invoke (works under NativeAOT).
    [DllImport("libc", SetLastError = true)] static extern int open(string path, int flags);
    [DllImport("libc", SetLastError = true)] static extern long pwrite(int fd, byte[] buf, nuint count, long off);
    [DllImport("libc", SetLastError = true)] static extern long pread(int fd, byte[] buf, nuint count, long off);
    [DllImport("libc", SetLastError = true)] static extern int fsync(int fd);
    const int O_RDWR = 2;
    static int disk = -1;

    static void Snapshot() {
        if (disk < 0) return;
        byte[] buf = new byte[512];
        BitConverter.GetBytes((uint)0x53464244).CopyTo(buf, 0);
        BitConverter.GetBytes((uint)applied).CopyTo(buf, 4);
        for (int i = 0; i < N; i++) BitConverter.GetBytes(bal[i]).CopyTo(buf, 8 + i * 8);
        pwrite(disk, buf, 512, 0);
        fsync(disk);
    }
    static bool Apply(int frm, int to, long amt) {
        lock (mu) {
            if (bal[frm] >= amt) {
                bal[frm] -= amt; bal[to] += amt; applied++;
                if (applied % SNAPIVL == 0) Snapshot();
                return true;
            }
            failed++; return false;
        }
    }
    static void Handle(TcpClient c) {
        try {
            var s = c.GetStream();
            var r = new StreamReader(s, Encoding.ASCII);
            var w = new StreamWriter(s, Encoding.ASCII) { AutoFlush = true };
            string line;
            while ((line = r.ReadLine()) != null) {
                var p = line.Split(' ');
                bool ok = Apply(int.Parse(p[0]), int.Parse(p[1]), long.Parse(p[2]));
                w.Write(ok ? "1\n" : "0\n");
            }
        } catch { }
        c.Close();
    }
    static void Main() {
        Console.WriteLine("CSVC-START csharp");
        disk = open("/dev/vda", O_RDWR);   // < 0 if unavailable
        if (disk < 0) Console.Error.WriteLine("DISK-OPEN-FAIL errno=" + Marshal.GetLastWin32Error());
        for (int i = 0; i < N; i++) bal[i] = START;
        var listener = new TcpListener(IPAddress.Loopback, PORT);
        listener.Start();
        new Thread(() => { try { while (true) { var c = listener.AcceptTcpClient(); new Thread(() => Handle(c)).Start(); } } catch { } }) { IsBackground = true }.Start();
        var threads = new Thread[CLIENTS];
        for (int c = 0; c < CLIENTS; c++) {
            int idx = c;
            threads[c] = new Thread(() => {
                long state = (idx + 1) * 1000 + 1;
                try {
                    var cl = new TcpClient();
                    cl.Connect(IPAddress.Loopback, PORT);
                    var s = cl.GetStream();
                    var r = new StreamReader(s, Encoding.ASCII);
                    var w = new StreamWriter(s, Encoding.ASCII) { AutoFlush = true };
                    for (int i = 0; i < PERCLIENT; i++) {
                        state = (state * 16807) % 2147483647; long frm = state % N;
                        state = (state * 16807) % 2147483647; long to = state % N;
                        state = (state * 16807) % 2147483647; long amt = state % 90 + 10;
                        w.Write($"{frm} {to} {amt}\n");
                        if (r.ReadLine() == null) break;
                    }
                    cl.Close();
                } catch { }
            });
            threads[c].Start();
        }
        foreach (var t in threads) t.Join();
        Snapshot();
        long total = 0;
        for (int i = 0; i < N; i++) total += bal[i];
        byte[] mb = new byte[N * 8];
        for (int i = 0; i < N; i++) BitConverter.GetBytes(bal[i]).CopyTo(mb, i * 8);
        ulong h = 0xcbf29ce484222325UL;
        foreach (byte b in mb) h = (h ^ b) * 0x100000001b3UL;
        long disktotal = -1;
        if (disk >= 0) {
            var rb = new byte[512];
            pread(disk, rb, 512, 0);
            if (BitConverter.ToUInt32(rb, 0) == 0x53464244) {
                disktotal = 0;
                for (int i = 0; i < N; i++) disktotal += BitConverter.ToInt64(rb, 8 + i * 8);
            }
        }
        Console.WriteLine($"CSVC-APPLIED {applied} CSVC-FAILED {failed}");
        Console.WriteLine($"INV-CONSERVED {(total == N * START).ToString().ToLower()} (total={total} want=2400)");
        Console.WriteLine($"INV-DISK-MATCH {(disktotal == total).ToString().ToLower()} (diskTotal={disktotal})");
        Console.WriteLine($"CSVC-HASH {h:x16}");
        Console.WriteLine("CSVC-EXIT 0");
    }
}
