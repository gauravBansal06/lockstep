// Distributed bank-service — Go (goroutines + net TCP), the concurrent main.go SPLIT
// into two roles over a real network. See ../c/distsvc.c.
//   distsvc server            → node B: bind 0.0.0.0:9000, apply transfers, invariant
//   distsvc client <peer_ip>  → node A: 6 workers × 500 requests to the server
package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"net"
	"os"
	"sync"
	"time"
)

const (
	N         = 16
	START     = 150
	CLIENTS   = 6
	PERCLIENT = 500
)

var (
	mu        sync.Mutex
	bal       [N]int64
	applied   int
	failed    int
	doneConns int
)

func apply(from, to int, amt int64) bool {
	mu.Lock()
	defer mu.Unlock()
	if bal[from] >= amt {
		bal[from] -= amt
		bal[to] += amt
		applied++
		return true
	}
	failed++
	return false
}

func handle(c net.Conn) {
	defer func() {
		c.Close()
		mu.Lock()
		doneConns++
		mu.Unlock()
	}()
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

func runServer() {
	fmt.Println("CSVC-START go-server")
	for i := range bal {
		bal[i] = START
	}
	ln, err := net.Listen("tcp", "0.0.0.0:9000")
	if err != nil {
		fmt.Println("CSVC-LISTEN-ERR", err)
		return
	}
	fmt.Println("CSVC-LISTEN 9000")
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go handle(c)
		}
	}()
	for i := 0; i < 4000; i++ { // ~40s virtual-time cap so a partition can't hang
		mu.Lock()
		d := doneConns
		mu.Unlock()
		if d >= CLIENTS {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	mu.Lock()
	var total int64
	for i := 0; i < N; i++ {
		total += bal[i]
	}
	memBuf := make([]byte, N*8)
	for i := 0; i < N; i++ {
		binary.LittleEndian.PutUint64(memBuf[i*8:], uint64(bal[i]))
	}
	mu.Unlock()
	fmt.Printf("CSVC-APPLIED %d CSVC-FAILED %d\n", applied, failed)
	fmt.Printf("INV-CONSERVED %v (total=%d want=2400)\n", total == int64(N*START), total)
	fmt.Printf("CSVC-HASH %016x\n", fnv(memBuf))
	fmt.Println("CSVC-EXIT 0")
}

func runClient(peer string) {
	fmt.Println("CSVC-START go-client")
	addr := peer + ":9000"
	var wg sync.WaitGroup
	for c := 0; c < CLIENTS; c++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			state := int64((idx+1)*1000 + 1)
			next := func() int64 { state = (state * 16807) % 2147483647; return state }
			var conn net.Conn
			var err error
			for r := 0; r < 80; r++ { // the server may still be booting
				conn, err = net.Dial("tcp", addr)
				if err == nil {
					break
				}
				time.Sleep(100 * time.Millisecond)
			}
			if err != nil {
				return
			}
			defer conn.Close()
			rd := bufio.NewReader(conn)
			w := bufio.NewWriter(conn)
			for i := 0; i < PERCLIENT; i++ {
				from := next() % N
				to := next() % N
				amt := next()%90 + 10
				fmt.Fprintf(w, "%d %d %d\n", from, to, amt)
				w.Flush()
				if _, e := rd.ReadString('\n'); e != nil {
					break
				}
			}
		}(c)
	}
	wg.Wait()
	fmt.Println("CSVC-CLIENT-DONE")
}

func main() {
	if len(os.Args) >= 2 && os.Args[1] == "server" {
		runServer()
	} else if len(os.Args) >= 3 && os.Args[1] == "client" {
		runClient(os.Args[2])
	} else {
		fmt.Println("usage: distsvc server | distsvc client <peer_ip>")
		os.Exit(1)
	}
}
