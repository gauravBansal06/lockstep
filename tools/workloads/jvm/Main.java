public class Main {
    public static void main(String[] args) {
        System.out.println("JVM-START");
        final long FNV = 1099511628211L;
        long h = -3750763034362895579L; // FNV-1a 64-bit offset basis
        // 1) integer + IEEE-754 float mix (JDK17 is strictfp => deterministic)
        double acc = 0.0;
        for (int i = 1; i <= 200000; i++) {
            acc += Math.sqrt((double) i) * 1.0000001;
            long v = (long) (acc * 1000.0) ^ (i * 2654435761L);
            h = (h ^ (v & 0xff)) * FNV;
            h = (h ^ ((v >>> 13) & 0xff)) * FNV;
        }
        // 2) deterministic PRNG fill + sort (exercises more JIT paths)
        int[] a = new int[2000];
        long s = 0x9E3779B97F4A7C15L;
        for (int i = 0; i < a.length; i++) {
            s = s * 6364136223846793005L + 1442695040888963407L;
            a[i] = (int) (s >>> 33);
        }
        java.util.Arrays.sort(a);
        for (int x : a) h = (h ^ (x & 0xff)) * FNV;
        // 3) string building / char ops
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 1000; i++) sb.append(i).append(',');
        String str = sb.toString();
        for (int i = 0; i < str.length(); i++) h = (h ^ str.charAt(i)) * FNV;
        System.out.printf("JVM-HASH %016x%n", h);
        System.out.println("JVM-EXIT 0");
    }
}
