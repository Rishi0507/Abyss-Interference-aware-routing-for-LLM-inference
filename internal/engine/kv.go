package engine

type block struct {
	hash       uint64
	ref        int32
	prev, next *block
	inLRU      bool
}

type kvStore interface {
	match(r *Request, maxBlocks int) int
	matchHost(r *Request, from, maxBlocks int) int
	acquire(r *Request, n int)
	acquireHost(r *Request, from, n int)
	alloc(pages int) bool
	publish(r *Request, idx int)
	release(r *Request)
	freePages() int
	usedPages() int
	cachedPages() int
	totalPages() int
}

type cacheKV struct {
	total, used, blockPages int
	caching                 bool
	blocks                  map[uint64]*block
	head, tail              *block
	lruLen                  int
	evictions               int64
	host                    *hostTier
	epoch                   uint64
	onEvict                 func(hash uint64)
	onStore                 func(hash uint64)
}

func newCacheKV(totalPages, blockPages int, caching bool) *cacheKV {
	return &cacheKV{total: totalPages, blockPages: blockPages, caching: caching, blocks: make(map[uint64]*block, 1024)}
}

func (c *cacheKV) match(r *Request, maxBlocks int) int {
	if !c.caching {
		return 0
	}
	n := 0
	for n < maxBlocks && n < len(r.Hashes) {
		if _, ok := c.blocks[r.Hashes[n]]; !ok {
			break
		}
		n++
	}
	return n
}

func (c *cacheKV) acquire(r *Request, n int) {
	for i := r.nShared; i < n; i++ {
		b := c.blocks[r.Hashes[i]]
		if b.ref == 0 {
			c.unlink(b)
		}
		b.ref++
		r.shared = append(r.shared, b)
	}
	r.nShared = n
}

func (c *cacheKV) alloc(pages int) bool {
	if pages <= 0 {
		return true
	}
	if c.total-c.used+c.lruLen*c.blockPages < pages {
		return false
	}
	for c.total-c.used < pages {
		c.evict()
	}
	c.used += pages
	return true
}

func (c *cacheKV) evict() {
	b := c.head
	c.unlink(b)
	delete(c.blocks, b.hash)
	c.used -= c.blockPages
	c.evictions++
	c.epoch++
	if c.host != nil {
		c.host.put(b.hash)
	}
	if c.onEvict != nil {
		c.onEvict(b.hash)
	}
}

func (c *cacheKV) matchHost(r *Request, from, maxBlocks int) int {
	if c.host == nil {
		return 0
	}
	n := 0
	for from+n < maxBlocks && from+n < len(r.Hashes) && c.host.has(r.Hashes[from+n]) {
		n++
	}
	return n
}

func (c *cacheKV) acquireHost(r *Request, from, n int) {
	for i := from; i < from+n; i++ {
		h := r.Hashes[i]
		b := &block{hash: h, ref: 1}
		c.blocks[h] = b
		c.epoch++
		r.shared = append(r.shared, b)
		c.host.touch(h)
	}
	r.nShared = from + n
}

func (c *cacheKV) publish(r *Request, idx int) {
	r.pages -= c.blockPages
	if !c.caching {
		r.ownPages += c.blockPages
		r.nShared++
		return
	}
	h := r.Hashes[idx]
	if b, ok := c.blocks[h]; ok {
		if b.ref == 0 {
			c.unlink(b)
		}
		b.ref++
		c.used -= c.blockPages
		r.shared = append(r.shared, b)
	} else {
		b := &block{hash: h, ref: 1}
		c.blocks[h] = b
		c.epoch++
		r.shared = append(r.shared, b)
		if c.onStore != nil {
			c.onStore(h)
		}
	}
	r.nShared++
}

func (c *cacheKV) release(r *Request) {
	c.used -= r.pages + r.ownPages
	r.pages, r.ownPages = 0, 0
	for i := len(r.shared) - 1; i >= 0; i-- {
		b := r.shared[i]
		b.ref--
		if b.ref == 0 {
			c.pushBack(b)
		}
	}
	r.shared = r.shared[:0]
	r.nShared = 0
}

func (c *cacheKV) unlink(b *block) {
	if !b.inLRU {
		return
	}
	if b.prev != nil {
		b.prev.next = b.next
	} else {
		c.head = b.next
	}
	if b.next != nil {
		b.next.prev = b.prev
	} else {
		c.tail = b.prev
	}
	b.prev, b.next, b.inLRU = nil, nil, false
	c.lruLen--
}

func (c *cacheKV) pushBack(b *block) {
	b.prev, b.next = c.tail, nil
	if c.tail != nil {
		c.tail.next = b
	} else {
		c.head = b
	}
	c.tail = b
	b.inLRU = true
	c.lruLen++
}

func (c *cacheKV) freePages() int   { return c.total - c.used }
func (c *cacheKV) usedPages() int   { return c.used }
func (c *cacheKV) cachedPages() int { return c.lruLen * c.blockPages }
func (c *cacheKV) totalPages() int  { return c.total }

func (c *cacheKV) evictionOrder(fn func(hash uint64) bool) {
	for b := c.head; b != nil; b = b.next {
		if !fn(b.hash) {
			return
		}
	}
}

func (c *cacheKV) contains(h uint64) bool {
	_, ok := c.blocks[h]
	return ok
}

type poolKV struct {
	total, used, blockPages int
}

func (p *poolKV) match(r *Request, maxBlocks int) int {
	if r.HitHint < maxBlocks {
		return r.HitHint
	}
	return maxBlocks
}

func (p *poolKV) acquire(r *Request, n int) { r.nShared = n }

func (p *poolKV) matchHost(r *Request, from, maxBlocks int) int {
	if from != r.HitHint {
		return 0
	}
	return min(r.HostHint, max(maxBlocks-from, 0))
}

func (p *poolKV) acquireHost(r *Request, from, n int) {
	r.ownPages += n * p.blockPages
	r.nShared = from + n
}

func (p *poolKV) alloc(pages int) bool {
	if pages <= 0 {
		return true
	}
	if p.total-p.used < pages {
		return false
	}
	p.used += pages
	return true
}

func (p *poolKV) publish(r *Request, idx int) {
	r.pages -= p.blockPages
	r.ownPages += p.blockPages
	r.nShared++
}

func (p *poolKV) release(r *Request) {
	p.used -= r.pages + r.ownPages
	r.pages, r.ownPages = 0, 0
	r.nShared = 0
}

func (p *poolKV) freePages() int   { return p.total - p.used }
func (p *poolKV) usedPages() int   { return p.used }
func (p *poolKV) cachedPages() int { return 0 }
func (p *poolKV) totalPages() int  { return p.total }

type hostTier struct {
	cap        int
	n          int
	blocks     map[uint64]*block
	head, tail *block
	evictions  int64
}

func newHostTier(capBlocks int) *hostTier {
	return &hostTier{cap: max(capBlocks, 1), blocks: make(map[uint64]*block, 1024)}
}

func (h *hostTier) has(x uint64) bool {
	_, ok := h.blocks[x]
	return ok
}

func (h *hostTier) touch(x uint64) {
	if b, ok := h.blocks[x]; ok {
		h.unlink(b)
		h.pushBack(b)
	}
}

func (h *hostTier) put(x uint64) {
	if b, ok := h.blocks[x]; ok {
		h.unlink(b)
		h.pushBack(b)
		return
	}
	for h.n >= h.cap && h.head != nil {
		old := h.head
		h.unlink(old)
		delete(h.blocks, old.hash)
		h.evictions++
	}
	b := &block{hash: x}
	h.blocks[x] = b
	h.pushBack(b)
}

func (h *hostTier) order(fn func(hash uint64) bool) {
	for b := h.head; b != nil; b = b.next {
		if !fn(b.hash) {
			return
		}
	}
}

func (h *hostTier) unlink(b *block) {
	if b.prev != nil {
		b.prev.next = b.next
	} else {
		h.head = b.next
	}
	if b.next != nil {
		b.next.prev = b.prev
	} else {
		h.tail = b.prev
	}
	b.prev, b.next = nil, nil
	h.n--
}

func (h *hostTier) pushBack(b *block) {
	b.prev, b.next = h.tail, nil
	if h.tail != nil {
		h.tail.next = b
	} else {
		h.head = b
	}
	h.tail = b
	h.n++
}

func (c *cacheKV) hints(r *Request) (int, int) {
	if r.hintEpoch == c.epoch+1 {
		return r.HitHint, r.HostHint
	}
	mb := (r.Target - 1) / r.blockTokens()
	a := c.match(r, mb)
	b := c.matchHost(r, a, mb)
	r.HitHint, r.HostHint, r.hintEpoch = a, b, c.epoch+1
	return a, b
}
