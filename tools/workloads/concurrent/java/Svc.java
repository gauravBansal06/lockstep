// Concurrent stateful service — Java (ServerSocket + threads + ReentrantLock). See ../SPEC.md.
import java.io.*;
import java.net.*;
import java.nio.*;
import java.util.concurrent.locks.*;

public class Svc {
    static final int N = 16, START = 150, CLIENTS = 6, PERCLIENT = 500, SNAPIVL = 137, PORT = 9000;
    static long[] bal = new long[N];
    static int applied = 0, failed = 0;
    static final ReentrantLock lock = new ReentrantLock();
    static RandomAccessFile disk = null;

    static void snapshot() throws IOException {
        if (disk == null) return;
        ByteBuffer b = ByteBuffer.allocate(512).order(ByteOrder.LITTLE_ENDIAN);
        b.putInt(0x53464244); b.putInt(applied);
        for (int i = 0; i < N; i++) b.putLong(bal[i]);
        disk.seek(0); disk.write(b.array()); disk.getFD().sync();
    }
    static boolean apply(int frm, int to, long amt) throws IOException {
        lock.lock();
        try {
            if (bal[frm] >= amt) {
                bal[frm] -= amt; bal[to] += amt; applied++;
                if (applied % SNAPIVL == 0) snapshot();
                return true;
            }
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
    }
    public static void main(String[] a) throws Exception {
        System.out.println("CSVC-START java");
        try { disk = new RandomAccessFile("/dev/vda", "rw"); } catch (Exception e) { disk = null; }
        for (int i = 0; i < N; i++) bal[i] = START;
        ServerSocket ss = new ServerSocket();
        ss.bind(new InetSocketAddress("127.0.0.1", PORT));
        Thread accept = new Thread(() -> {
            try { while (true) { Socket s = ss.accept(); new Thread(() -> handle(s)).start(); } } catch (Exception e) {}
        });
        accept.setDaemon(true); accept.start();
        Thread[] cl = new Thread[CLIENTS];
        for (int c = 0; c < CLIENTS; c++) {
            final int idx = c;
            cl[c] = new Thread(() -> {
                long state = (long) (idx + 1) * 1000 + 1;
                try (Socket s = new Socket("127.0.0.1", PORT);
                     BufferedReader r = new BufferedReader(new InputStreamReader(s.getInputStream()));
                     BufferedWriter w = new BufferedWriter(new OutputStreamWriter(s.getOutputStream()))) {
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
        snapshot();
        long total = 0;
        for (int i = 0; i < N; i++) total += bal[i];
        ByteBuffer mb = ByteBuffer.allocate(N * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < N; i++) mb.putLong(bal[i]);
        long h = 0xcbf29ce484222325L;
        for (byte bb : mb.array()) h = (h ^ (bb & 0xff)) * 0x100000001b3L;
        long disktotal = -1;
        if (disk != null) {
            byte[] rb = new byte[512]; disk.seek(0); disk.readFully(rb);
            ByteBuffer rbb = ByteBuffer.wrap(rb).order(ByteOrder.LITTLE_ENDIAN);
            if (rbb.getInt(0) == 0x53464244) { disktotal = 0; for (int i = 0; i < N; i++) disktotal += rbb.getLong(8 + i * 8); }
        }
        System.out.printf("CSVC-APPLIED %d CSVC-FAILED %d%n", applied, failed);
        System.out.printf("INV-CONSERVED %b (total=%d want=2400)%n", total == N * START, total);
        System.out.printf("INV-DISK-MATCH %b (diskTotal=%d)%n", disktotal == total, disktotal);
        System.out.printf("CSVC-HASH %016x%n", h);
        System.out.println("CSVC-EXIT 0");
        System.exit(0);
    }
}
