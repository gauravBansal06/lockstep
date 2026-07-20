# Distributed bank-service — Ruby (TCPServer + Thread + Mutex), the concurrent svc.rb
# SPLIT into two roles over a real network. See ../c/distsvc.c.
#   distsvc.rb server            → node B: bind 0.0.0.0:9000, apply transfers, invariant
#   distsvc.rb client <peer_ip>  → node A: 6 workers × 500 requests to the server
require 'socket'
N = 16; START = 150; CLIENTS = 6; PERCLIENT = 500; PORT = 9000
$bal = Array.new(N, START); $applied = 0; $failed = 0; $done = 0; $mu = Mutex.new

def apply(frm, to, amt)
  $mu.synchronize do
    if $bal[frm] >= amt
      $bal[frm] -= amt; $bal[to] += amt; $applied += 1
      true
    else
      $failed += 1; false
    end
  end
end

def run_server
  print "CSVC-START ruby-server\n"; $stdout.flush
  srv = TCPServer.new('0.0.0.0', PORT)
  print "CSVC-LISTEN 9000\n"; $stdout.flush
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
        $mu.synchronize { $done += 1 }
      end
    end
  end
  4000.times do             # ~40s virtual-time cap so a partition can't hang
    break if $mu.synchronize { $done } >= CLIENTS
    sleep 0.01
  end
  total = $bal.sum
  mb = $bal.pack('q<*')
  h = 0xcbf29ce484222325; mask = 0xFFFFFFFFFFFFFFFF
  mb.each_byte { |b| h = ((h ^ b) * 0x100000001b3) & mask }
  printf("CSVC-APPLIED %d CSVC-FAILED %d\n", $applied, $failed)
  printf("INV-CONSERVED %s (total=%d want=2400)\n", (total == N * START).to_s, total)
  printf("CSVC-HASH %016x\n", h)
  print "CSVC-EXIT 0\n"; $stdout.flush
end

def run_client(peer)
  print "CSVC-START ruby-client\n"; $stdout.flush
  threads = (0...CLIENTS).map do |c|
    Thread.new(c) do |idx|
      state = (idx + 1) * 1000 + 1
      nxt = lambda { state = (state * 16807) % 2147483647; state }
      conn = nil
      80.times do           # the server may still be booting
        begin; conn = TCPSocket.new(peer, PORT); break; rescue StandardError; sleep 0.1; end
      end
      next if conn.nil?
      PERCLIENT.times do
        f = nxt.call % N; t = nxt.call % N; a = nxt.call % 90 + 10
        conn.write("#{f} #{t} #{a}\n")
        break if conn.gets.nil?
      end
      conn.close rescue nil
    end
  end
  threads.each(&:join)
  print "CSVC-CLIENT-DONE\n"; $stdout.flush
end

if ARGV[0] == 'server'
  run_server
elsif ARGV[0] == 'client' && ARGV[1]
  run_client(ARGV[1])
else
  print "usage: distsvc.rb server | distsvc.rb client <peer_ip>\n"
  exit 1
end
