// Distributed bank-service — C# (.NET NativeAOT, TcpListener + threads + lock), the
// concurrent Svc.cs SPLIT into two roles over a real network. See ../c/distsvc.c.
//   distsvc server            → node B: bind 0.0.0.0:9000, apply transfers, invariant
//   distsvc client <peer_ip>  → node A: 6 workers × 500 requests to the server
using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

class Distsvc {
    const int N = 16, START = 150, CLIENTS = 6, PERCLIENT = 500, PORT = 9000;
    static long[] bal = new long[N];
    static int applied = 0, failed = 0, doneConns = 0;
    static readonly object mu = new object();

    static bool Apply(int frm, int to, long amt) {
        lock (mu) {
            if (bal[frm] >= amt) { bal[frm] -= amt; bal[to] += amt; applied++; return true; }
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
        lock (mu) { doneConns++; }
    }
    static void RunServer() {
        Console.WriteLine("CSVC-START csharp-server");
        for (int i = 0; i < N; i++) bal[i] = START;
        var listener = new TcpListener(IPAddress.Any, PORT);
        listener.Start();
        Console.WriteLine("CSVC-LISTEN 9000");
        new Thread(() => { try { while (true) { var c = listener.AcceptTcpClient(); new Thread(() => Handle(c)).Start(); } } catch { } }) { IsBackground = true }.Start();
        for (int i = 0; i < 4000; i++) {   // ~40s virtual-time cap so a partition can't hang
            int d; lock (mu) { d = doneConns; }
            if (d >= CLIENTS) break;
            Thread.Sleep(10);
        }
        long total = 0;
        for (int i = 0; i < N; i++) total += bal[i];
        byte[] mb = new byte[N * 8];
        for (int i = 0; i < N; i++) BitConverter.GetBytes(bal[i]).CopyTo(mb, i * 8);
        ulong h = 0xcbf29ce484222325UL;
        foreach (byte b in mb) h = (h ^ b) * 0x100000001b3UL;
        Console.WriteLine($"CSVC-APPLIED {applied} CSVC-FAILED {failed}");
        Console.WriteLine($"INV-CONSERVED {(total == N * START).ToString().ToLower()} (total={total} want=2400)");
        Console.WriteLine($"CSVC-HASH {h:x16}");
        Console.WriteLine("CSVC-EXIT 0");
    }
    static void RunClient(string peer) {
        Console.WriteLine("CSVC-START csharp-client");
        var threads = new Thread[CLIENTS];
        for (int c = 0; c < CLIENTS; c++) {
            int idx = c;
            threads[c] = new Thread(() => {
                long state = (idx + 1) * 1000 + 1;
                TcpClient cl = null;
                for (int r = 0; r < 80; r++) {   // the server may still be booting
                    try { cl = new TcpClient(); cl.Connect(peer, PORT); break; }
                    catch { cl = null; Thread.Sleep(100); }
                }
                if (cl == null) return;
                try {
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
        Console.WriteLine("CSVC-CLIENT-DONE");
    }
    static int Main(string[] args) {
        if (args.Length >= 1 && args[0] == "server") RunServer();
        else if (args.Length >= 2 && args[0] == "client") RunClient(args[1]);
        else { Console.WriteLine("usage: distsvc server | distsvc client <peer_ip>"); return 1; }
        return 0;
    }
}
