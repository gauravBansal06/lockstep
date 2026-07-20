# Concurrent stateful service — Ruby (TCPServer + Thread + Mutex). See ../SPEC.md.
require 'socket'
N = 16; START = 150; CLIENTS = 6; PERCLIENT = 500; SNAPIVL = 137; PORT = 9000
$bal = Array.new(N, START); $applied = 0; $failed = 0; $mu = Mutex.new; $disk = nil

def snapshot
  return if $disk.nil?
  buf = [0x53464244, $applied].pack('VV') + $bal.pack('q<*')
  buf += "\0" * (512 - buf.bytesize)
  $disk.seek(0); $disk.write(buf); $disk.fsync
end

def apply(frm, to, amt)
  $mu.synchronize do
    if $bal[frm] >= amt
      $bal[frm] -= amt; $bal[to] += amt; $applied += 1
      snapshot if $applied % SNAPIVL == 0
      true
    else
      $failed += 1; false
    end
  end
end

print "CSVC-START ruby\n"
begin; $disk = File.open('/dev/vda', 'r+b'); rescue StandardError; $disk = nil; end
srv = TCPServer.new('127.0.0.1', PORT)
Thread.new do
  loop do
    c = srv.accept
    Thread.new(c) do |conn|
      begin
        while (line = conn.gets)
          f, t, a = line.split.map(&:to_i)
          conn.write(apply(f, t, a) ? "1\n" : "0\n")
        end
      rescue StandardError
      end
      conn.close rescue nil
    end
  end
end

threads = (0...CLIENTS).map do |c|
  Thread.new(c) do |idx|
    state = (idx + 1) * 1000 + 1
    nxt = lambda { state = (state * 16807) % 2147483647; state }
    conn = TCPSocket.new('127.0.0.1', PORT)
    PERCLIENT.times do
      f = nxt.call % N; t = nxt.call % N; a = nxt.call % 90 + 10
      conn.write("#{f} #{t} #{a}\n")
      break if conn.gets.nil?
    end
    conn.close rescue nil
  end
end
threads.each(&:join)

snapshot
total = $bal.sum
mb = $bal.pack('q<*')
h = 0xcbf29ce484222325; mask = 0xFFFFFFFFFFFFFFFF
mb.each_byte { |b| h = ((h ^ b) * 0x100000001b3) & mask }
disktotal = -1
if $disk
  $disk.seek(0); rb = $disk.read(512)
  if rb && rb[0, 4].unpack1('V') == 0x53464244
    disktotal = rb[8, N * 8].unpack('q<*').sum
  end
end
printf("CSVC-APPLIED %d CSVC-FAILED %d\n", $applied, $failed)
printf("INV-CONSERVED %s (total=%d want=2400)\n", (total == N * START).to_s, total)
printf("INV-DISK-MATCH %s (diskTotal=%d)\n", (disktotal == total).to_s, disktotal)
printf("CSVC-HASH %016x\n", h)
print "CSVC-EXIT 0\n"
