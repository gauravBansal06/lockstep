// A real stateful "bank" microservice exercising memory + loopback network + disk I/O,
// with a conservation invariant. Concurrent clients fire HTTP transfers; the server moves
// money under a lock IFF the source has funds (non-commutative => order matters). State is
// periodically snapshotted to /dev/vda (virtio-blk) and read back to verify. Correct code =>
// the invariant (total conserved, no negative) holds under EVERY interleaving; different
// seeds explore different valid end-states. A race bug would violate it under some seeds.
package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"sync"
)

const (
	N        = 16      // accounts
	START    = 150    // per account
	CLIENTS  = 6
	PERCLIENT = 800
	SNAPIVL  = 137     // snapshot every N applied transfers
)

var (
	mu      sync.Mutex
	bal     [N]int64
	applied int
	failed  int
	disk    *os.File
)

func snapshotLocked() { // caller holds mu
	var buf [512]byte
	binary.LittleEndian.PutUint32(buf[0:], 0x5346_4244) // "DBFS" magic
	binary.LittleEndian.PutUint32(buf[4:], uint32(applied))
	for i := 0; i < N; i++ {
		binary.LittleEndian.PutUint64(buf[8+i*8:], uint64(bal[i]))
	}
	if disk != nil {
		disk.WriteAt(buf[:], 0)
		disk.Sync()
	}
}

func transfer(from, to int, amt int64) bool {
	mu.Lock()
	defer mu.Unlock()
	if bal[from] >= amt { // sufficiency check => non-commutative
		bal[from] -= amt
		bal[to] += amt
		applied++
		if applied%SNAPIVL == 0 {
			snapshotLocked()
		}
		return true
	}
	failed++
	return false
}

func fnv(b []byte) uint64 {
	h := uint64(14695981039346656037)
	for _, c := range b {
		h = (h ^ uint64(c)) * 1099511628211
	}
	return h
}

func main() {
	fmt.Println("STATE-START")
	if f, err := os.OpenFile("/dev/vda", os.O_RDWR, 0); err == nil {
		disk = f
	} else {
		fmt.Println("STATE-NODISK", err)
	}
	for i := range bal {
		bal[i] = START
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/transfer", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		from, _ := strconv.Atoi(q.Get("from"))
		to, _ := strconv.Atoi(q.Get("to"))
		amt, _ := strconv.ParseInt(q.Get("amt"), 10, 64)
		if transfer(from, to, amt) {
			io.WriteString(w, "ok")
		} else {
			io.WriteString(w, "insufficient")
		}
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "ok") })
	go http.ListenAndServe("127.0.0.1:8080", mux)
	for i := 0; i < 2000; i++ { // wait for server
		if resp, err := http.Get("http://127.0.0.1:8080/healthz"); err == nil {
			resp.Body.Close()
			break
		}
	}
	fmt.Println("SERVICE-UP")

	var wg sync.WaitGroup
	for c := 0; c < CLIENTS; c++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed)) // deterministic OP params; only ORDER varies
			for i := 0; i < PERCLIENT; i++ {
				from := rng.Intn(N)
				to := rng.Intn(N)
				amt := int64(rng.Intn(90) + 10)
				url := fmt.Sprintf("http://127.0.0.1:8080/transfer?from=%d&to=%d&amt=%d", from, to, amt)
				if resp, err := http.Post(url, "", nil); err == nil {
					io.Copy(io.Discard, resp.Body)
					resp.Body.Close()
				}
			}
		}(int64(c*7919 + 1))
	}
	wg.Wait()

	// final snapshot, then read it back from disk and verify
	mu.Lock()
	snapshotLocked()
	var total int64
	neg := false
	for i := 0; i < N; i++ {
		total += bal[i]
		if bal[i] < 0 {
			neg = true
		}
	}
	memBuf := make([]byte, N*8)
	for i := 0; i < N; i++ {
		binary.LittleEndian.PutUint64(memBuf[i*8:], uint64(bal[i]))
	}
	mu.Unlock()

	// disk read-back
	diskTotal := int64(-1)
	diskHash := uint64(0)
	if disk != nil {
		var rb [512]byte
		disk.ReadAt(rb[:], 0)
		if binary.LittleEndian.Uint32(rb[0:]) == 0x5346_4244 {
			dbuf := make([]byte, N*8)
			diskTotal = 0
			for i := 0; i < N; i++ {
				v := int64(binary.LittleEndian.Uint64(rb[8+i*8:]))
				diskTotal += v
				binary.LittleEndian.PutUint64(dbuf[i*8:], uint64(v))
			}
			diskHash = fnv(dbuf)
		}
	}

	fmt.Printf("STATE-APPLIED %d STATE-FAILED %d\n", applied, failed)
	fmt.Printf("INV-CONSERVED %v (total=%d want=%d)\n", total == int64(N*START), total, N*START)
	fmt.Printf("INV-NONNEG %v\n", !neg)
	fmt.Printf("INV-DISK-MATCH %v (diskTotal=%d)\n", diskTotal == total, diskTotal)
	fmt.Printf("STATE-HASH %016x\n", fnv(memBuf))
	fmt.Printf("DISK-HASH %016x\n", diskHash)
	fmt.Println("STATE-EXIT 0")
}
