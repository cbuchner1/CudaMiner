#pragma once
extern "C" { int gettimeofday(struct timeval *tv, struct timezone *tz); }
extern "C" { void usleep(__int64 usec); }
typedef __int64 useconds_t;
