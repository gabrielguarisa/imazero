#if !defined(_TEST_)
#define _TEST_

void zerar(void *const _ptr, const int length) {
  uint8_t *const ptr = (uint8_t *)_ptr;
  for (int i = 0; i != length; ++i)
    ptr[i] = 0;
}



#endif // _TEST_
