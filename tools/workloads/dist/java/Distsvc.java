// Distributed bank-service — Java (ServerSocket + threads + ReentrantLock), the concurrent
// Svc.java SPLIT into two roles over a real network. See ../c/distsvc.c.
//   distsvc server            → node B: bind 0.0.0.0:9000, apply transfers, invariant
//   distsvc client <peer_ip>  → node A: 6 workers × 500 requests to the server
import java.io.*;
import java.net.*;
import java.nio.*;
import java.util.concurrent.locks.*;

public class Distsvc {
    static final int N = 16, START = 150, CLIENTS = 6, PERCLIENT = 500, PORT = 9000;
    static long[] bal = new long[N];
    static int applied = 0, failed = 0, doneConns = 0;
    static final ReentrantLock lock = new ReentrantLock();

    static boolean apply(int frm, int to, long amt) {
        lock.lock();
        try {
            if (bal[frm] >= amt) { bal[frm] -= amt; bal[to] += amt; applied++; return true; }
            failed++; return false;
        } finally { lock.unlock(); }
    }
    static void handle(Socket s) {
        try (BufferedReader r = new BufferedReader(new InputStreamReader(s.getInputStream()));
             BufferedWriter w = new BufferedWriter(new OutputStreamWriter(s.getOutputStream()))) {
            String line;
            while ((line = r.readLine()) != null) {
                String[] p = line.split(" ");
                boolean ok = apply(Integer.parseInt(p[0]), Integer.parseInt(p[1]), Long.parseLong(p[2]));
                w.write(ok ? "1\n" : "0\n"); w.flush();
            }
        } catch (Exception e) {}
        lock.lock(); doneConns++; lock.unlock();
    }
    static void runServer() throws Exception {
        System.out.println("CSVC-START java-server");
        for (int i = 0; i < N; i++) bal[i] = START;
        ServerSocket ss = new ServerSocket();
        ss.setReuseAddress(true);
        ss.bind(new InetSocketAddress("0.0.0.0", PORT));
        System.out.println("CSVC-LISTEN 9000");
        Thread accept = new Thread(() -> {
            try { while (true) { Socket s = ss.accept(); new Thread(() -> handle(s)).start(); } } catch (Exception e) {}
        });
        accept.setDaemon(true); accept.start();
        for (int i = 0; i < 4000; i++) {   // ~40s virtual-time cap so a partition can't hang
            lock.lock(); int d = doneConns; lock.unlock();
            if (d >= CLIENTS) break;
            Thread.sleep(10);
        }
        long total = 0;
        for (int i = 0; i < N; i++) total += bal[i];
        ByteBuffer mb = ByteBuffer.allocate(N * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < N; i++) mb.putLong(bal[i]);
        long h = 0xcbf29ce484222325L;
        for (byte bb : mb.array()) h = (h ^ (bb & 0xff)) * 0x100000001b3L;
        System.out.printf("CSVC-APPLIED %d CSVC-FAILED %d%n", applied, failed);
        System.out.printf("INV-CONSERVED %b (total=%d want=2400)%n", total == N * START, total);
        System.out.printf("CSVC-HASH %016x%n", h);
        System.out.println("CSVC-EXIT 0");
        System.exit(0);
    }
    static void runClient(String peer) throws Exception {
        System.out.println("CSVC-START java-client");
        Thread[] cl = new Thread[CLIENTS];
        for (int c = 0; c < CLIENTS; c++) {
            final int idx = c;
            cl[c] = new Thread(() -> {
                long state = (long) (idx + 1) * 1000 + 1;
                Socket s = null;
                for (int r = 0; r < 80; r++) {   // the server may still be booting
                    try { s = new Socket(peer, PORT); break; } catch (Exception e) {
                        try { Thread.sleep(100); } catch (Exception ie) {}
                    }
                }
                if (s == null) return;
                try (Socket sock = s;
                     BufferedReader r = new BufferedReader(new InputStreamReader(sock.getInputStream()));
                     BufferedWriter w = new BufferedWriter(new OutputStreamWriter(sock.getOutputStream()))) {
                    for (int i = 0; i < PERCLIENT; i++) {
                        state = (state * 16807) % 2147483647; long frm = state % N;
                        state = (state * 16807) % 2147483647; long to = state % N;
                        state = (state * 16807) % 2147483647; long amt = state % 90 + 10;
                        w.write(frm + " " + to + " " + amt + "\n"); w.flush();
                        if (r.readLine() == null) break;
                    }
                } catch (Exception e) {}
            });
            cl[c].start();
        }
        for (Thread t : cl) t.join();
        System.out.println("CSVC-CLIENT-DONE");
        System.exit(0);
    }
    public static void main(String[] a) throws Exception {
        if (a.length >= 1 && a[0].equals("server")) runServer();
        else if (a.length >= 2 && a[0].equals("client")) runClient(a[1]);
        else { System.out.println("usage: Distsvc server | Distsvc client <peer_ip>"); System.exit(1); }
    }
}
