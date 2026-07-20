// Concurrent stateful service — Go (goroutines + net TCP). See ../SPEC.md.
package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"net"
	"os"
	"sync"
)

const (
	N         = 16
	START     = 150
	CLIENTS   = 6
	PERCLIENT = 500
	SNAPIVL   = 137
	PORT      = "127.0.0.1:9000"
)

var (
	mu      sync.Mutex
	bal     [N]int64
	applied int
	failed  int
	disk    *os.File
)

func snapshotLocked() {
	var buf [512]byte
	binary.LittleEndian.PutUint32(buf[0:], 0x53464244)
	binary.LittleEndian.PutUint32(buf[4:], uint32(applied))
	for i := 0; i < N; i++ {
		binary.LittleEndian.PutUint64(buf[8+i*8:], uint64(bal[i]))
	}
	if disk != nil {
		disk.WriteAt(buf[:], 0)
		disk.Sync()
	}
}

func apply(from, to int, amt int64) bool {
	mu.Lock()
	defer mu.Unlock()
	if bal[from] >= amt {
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

func handle(c net.Conn) {
	defer c.Close()
	r := bufio.NewReader(c)
	w := bufio.NewWriter(c)
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			break
		}
		var from, to int
		var amt int64
		if _, e := fmt.Sscanf(line, "%d %d %d", &from, &to, &amt); e != nil {
			break
		}
		if apply(from, to, amt) {
			w.WriteString("1\n")
		} else {
			w.WriteString("0\n")
		}
		w.Flush()
	}
}

func fnv(b []byte) uint64 {
	h := uint64(0xcbf29ce484222325)
	for _, c := range b {
		h = (h ^ uint64(c)) * 0x100000001b3
	}
	return h
}

func main() {
	fmt.Println("CSVC-START go")
	if f, err := os.OpenFile("/dev/vda", os.O_RDWR, 0); err == nil {
		disk = f
	}
	for i := range bal {
		bal[i] = START
	}
	ln, err := net.Listen("tcp", PORT)
	if err != nil {
		fmt.Println("CSVC-LISTEN-ERR", err)
		return
	}
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go handle(c)
		}
	}()

	var wg sync.WaitGroup
	for c := 0; c < CLIENTS; c++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			state := int64((idx+1)*1000 + 1)
			next := func() int64 { state = (state * 16807) % 2147483647; return state }
			conn, err := net.Dial("tcp", PORT)
			if err != nil {
				return
			}
			defer conn.Close()
			r := bufio.NewReader(conn)
			w := bufio.NewWriter(conn)
			for i := 0; i < PERCLIENT; i++ {
				from := next() % N
				to := next() % N
				amt := next()%90 + 10
				fmt.Fprintf(w, "%d %d %d\n", from, to, amt)
				w.Flush()
				if _, e := r.ReadString('\n'); e != nil {
					break
				}
			}
		}(c)
	}
	wg.Wait()

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

	diskTotal := int64(-1)
	if disk != nil {
		var rb [512]byte
		if _, err := disk.ReadAt(rb[:], 0); err == nil && binary.LittleEndian.Uint32(rb[0:]) == 0x53464244 {
			diskTotal = 0
			for i := 0; i < N; i++ {
				diskTotal += int64(binary.LittleEndian.Uint64(rb[8+i*8:]))
			}
		}
	}

	fmt.Printf("CSVC-APPLIED %d CSVC-FAILED %d\n", applied, failed)
	fmt.Printf("INV-CONSERVED %v (total=%d want=2400)\n", total == int64(N*START), total)
	fmt.Printf("INV-DISK-MATCH %v (diskTotal=%d)\n", diskTotal == total, diskTotal)
	fmt.Printf("CSVC-HASH %016x\n", fnv(memBuf))
	_ = neg
	fmt.Println("CSVC-EXIT 0")
}
