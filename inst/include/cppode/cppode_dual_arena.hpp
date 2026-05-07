/*
 Thread-local bump arena for cppode::dual / cppode::dual2nd derivative storage.

 Design:
 - One thread-local arena per thread, growing slabs (initial 64 KiB, doubling).
 - LIFO scope-based reset via cppode::dual_arena::scope (RAII).
 - Bump-pointer alloc<T>(n): O(1) per allocation, no per-temporary heap call.
 - For trivially-destructible T (double, float, int, …) the alloc is a pure
   bump; no ctor/dtor bookkeeping. For non-trivially-destructible T (e.g.
   boost::multiprecision::cpp_dec_float, mpfr::mpreal) the alloc records a
   destructor entry; on scope rollback the recorded dtors are invoked LIFO.

 Usage (codegen wraps each RHS evaluation):

   void rhs(...) {
     cppode::dual_arena::scope guard;
     // ... allocate dual<double, 0> objects via .diff(idx, n_sens) ...
   }                       // <- guard destructor pops every alloc since ctor

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_DUAL_ARENA_HPP
#define CPPODE_DUAL_ARENA_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <vector>

namespace cppode {

class dual_arena {
public:
  static constexpr std::size_t initial_slab_bytes = 64u * 1024u;

  struct slab {
    char*       data = nullptr;
    std::size_t cap  = 0;
    std::size_t top  = 0;
    slab() = default;
    explicit slab(std::size_t bytes)
      : data(static_cast<char*>(std::malloc(bytes))), cap(bytes), top(0)
    {}
    slab(const slab&) = delete;
    slab& operator=(const slab&) = delete;
    slab(slab&& o) noexcept : data(o.data), cap(o.cap), top(o.top) { o.data=nullptr; o.cap=0; o.top=0; }
    slab& operator=(slab&& o) noexcept {
      if (this != &o) { std::free(data); data=o.data; cap=o.cap; top=o.top; o.data=nullptr; o.cap=0; o.top=0; }
      return *this;
    }
    ~slab() { std::free(data); }
  };

  using dtor_fn = void(*)(void*);
  struct dtor_entry { dtor_fn fn; void* obj; };

  std::vector<slab>       slabs_;
  std::size_t             active_ = 0;          // index of current slab
  std::vector<dtor_entry> dtors_;

  dual_arena() {
    slabs_.emplace_back(initial_slab_bytes);
  }

  // Bump allocation for trivially-destructible T (fast path).
  // Caller is responsible for writing into the returned memory.
  template<class T>
  T* alloc_trivial(std::size_t n) {
    static_assert(std::is_trivially_destructible_v<T>,
                  "alloc_trivial requires trivially-destructible T");
    return static_cast<T*>(bump(alignof(T), n * sizeof(T)));
  }

  // General alloc: default-constructs n elements of T and (if non-trivial)
  // records destructors for LIFO rollback.
  template<class T>
  T* alloc(std::size_t n) {
    void* raw = bump(alignof(T), n * sizeof(T));
    T* arr = static_cast<T*>(raw);
    if constexpr (std::is_trivially_destructible_v<T>) {
      // Trivial: skip default ctor (caller writes); skip dtor tracking.
      return arr;
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        ::new (static_cast<void*>(arr + i)) T();
        dtors_.push_back({ &dtor_thunk<T>, static_cast<void*>(arr + i) });
      }
      return arr;
    }
  }

  // RAII scope: snapshot active slab + top + dtor count on construction;
  // restore on destruction (calls recorded dtors in reverse).
  class scope {
    dual_arena& a_;
    std::size_t saved_active_;
    std::size_t saved_top_;
    std::size_t saved_dtors_;
  public:
    scope() noexcept
      : a_(arena()),
        saved_active_(a_.active_),
        saved_top_(a_.slabs_[a_.active_].top),
        saved_dtors_(a_.dtors_.size())
    {}
    scope(const scope&)            = delete;
    scope& operator=(const scope&) = delete;
    ~scope() noexcept {
      // Pop recorded dtors LIFO.
      while (a_.dtors_.size() > saved_dtors_) {
        auto& e = a_.dtors_.back();
        e.fn(e.obj);
        a_.dtors_.pop_back();
      }
      // Restore slab tops: any slab beyond saved_active_ is fully free; the
      // saved one rolls back to saved_top_. We keep the slabs alive for reuse.
      for (std::size_t i = a_.active_; i > saved_active_; --i) {
        a_.slabs_[i].top = 0;
      }
      a_.active_ = saved_active_;
      a_.slabs_[saved_active_].top = saved_top_;
    }
  };

  static dual_arena& arena() noexcept {
    thread_local dual_arena tls;
    return tls;
  }

private:
  void* bump(std::size_t align, std::size_t bytes) {
    slab* s = &slabs_[active_];
    std::size_t aligned = (s->top + (align - 1)) & ~(align - 1);
    if (aligned + bytes > s->cap) {
      grow(align, bytes);
      s = &slabs_[active_];
      aligned = (s->top + (align - 1)) & ~(align - 1);
    }
    void* p = s->data + aligned;
    s->top = aligned + bytes;
    return p;
  }

  void grow(std::size_t align, std::size_t bytes) {
    // Find an existing free slab past active_ that fits, or push a new one.
    std::size_t need = bytes + align;  // worst-case alignment slack
    for (std::size_t i = active_ + 1; i < slabs_.size(); ++i) {
      if (slabs_[i].cap >= need) {
        active_ = i;
        slabs_[i].top = 0;
        return;
      }
    }
    std::size_t prev_cap = slabs_.back().cap;
    std::size_t new_cap = prev_cap * 2;
    if (new_cap < need) new_cap = need;
    slabs_.emplace_back(new_cap);
    active_ = slabs_.size() - 1;
  }

  template<class T>
  static void dtor_thunk(void* p) noexcept(std::is_nothrow_destructible_v<T>) {
    static_cast<T*>(p)->~T();
  }
};

} // namespace cppode

#endif // CPPODE_DUAL_ARENA_HPP
