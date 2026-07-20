# Deterministic pure-compute Ruby (core only, --disable-gems, no require).
puts "RUBY-START"
srand(987654321)
acc = 0.0
i = 1
while i <= 200000
  acc += Math.sqrt(i) * 1.0000001 + (i % 97) / 7.0
  i += 1
end
arr = Array.new(5000) { rand(1_000_000_000) }
arr.sort!
sum = arr.inject(0) { |a, b| a + b }
blob = format("%.12e|%d|%d|%d", acc, sum, arr[0], arr[4999])
blob += "|" + arr[0, 200].join(",")
blob += "|" + arr[0, 1000].map { |x| x.to_s(16) }.join
mask = 0xFFFFFFFFFFFFFFFF
h = 14695981039346656037
blob.each_byte { |b| h = ((h ^ b) * 1099511628211) & mask }
puts format("RUBY-HASH %016x", h)
puts "RUBY-EXIT 0"
