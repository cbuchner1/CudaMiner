/*
 * Copyright 2010 Jeff Garzik
 * Copyright 2012-2013 pooler
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#define _GNU_SOURCE
#include "cpuminer-config.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <jansson.h>
#include <curl/curl.h>
#include <time.h>
#if defined(WIN32)
#include <winsock2.h>
#include <mstcpip.h>
#else
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#endif
#include "compat.h"
#include "miner.h"
#include "elist.h"

struct data_buffer {
	void		*buf;
	size_t		len;
};

struct upload_buffer {
	const void	*buf;
	size_t		len;
	size_t		pos;
};

struct header_info {
	char		*lp_path;
	char		*reason;
	char		*stratum_url;
};

struct tq_ent {
	void			*data;
	struct list_head	q_node;
};

struct thread_q {
	struct list_head	q;

	bool frozen;

	pthread_mutex_t		mutex;
	pthread_cond_t		cond;
};

void applog(int prio, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);

#ifdef HAVE_SYSLOG_H
	if (use_syslog) {
		va_list ap2;
		char *buf;
		int len;
		
		va_copy(ap2, ap);
		len = vsnprintf(NULL, 0, fmt, ap2) + 1;
		va_end(ap2);
		buf = alloca(len);
		if (vsnprintf(buf, len, fmt, ap) >= 0)
			syslog(prio, "%s", buf);
	}
#else
	if (0) {}
#endif
	else {
		char *f;
		int len;
		time_t now;
		struct tm tm, *tm_p;

		time(&now);

		pthread_mutex_lock(&applog_lock);
		tm_p = localtime(&now);
		memcpy(&tm, tm_p, sizeof(tm));
		pthread_mutex_unlock(&applog_lock);

		len = 40 + strlen(fmt) + 2;
		f = (char*)alloca(len);
		sprintf(f, "[%d-%02d-%02d %02d:%02d:%02d] %s\n",
			tm.tm_year + 1900,
			tm.tm_mon + 1,
			tm.tm_mday,
			tm.tm_hour,
			tm.tm_min,
			tm.tm_sec,
			fmt);
		pthread_mutex_lock(&applog_lock);
		vfprintf(stderr, f, ap);	/* atomic write to stderr */
		fflush(stderr);
		pthread_mutex_unlock(&applog_lock);
	}
	va_end(ap);
}

static void databuf_free(struct data_buffer *db)
{
	if (!db)
		return;

	free(db->buf);

	memset(db, 0, sizeof(*db));
}

static size_t all_data_cb(const void *ptr, size_t size, size_t nmemb,
			  void *user_data)
{
	struct data_buffer *db = (struct data_buffer *)user_data;
	size_t len = size * nmemb;
	size_t oldlen, newlen;
	void *newmem;
	static const unsigned char zero = 0;

	oldlen = db->len;
	newlen = oldlen + len;

	newmem = realloc(db->buf, newlen + 1);
	if (!newmem)
		return 0;

	db->buf = newmem;
	db->len = newlen;
	memcpy((char*)db->buf + oldlen, ptr, len);
	memcpy((char*)db->buf + newlen, &zero, 1);	/* null terminate */

	return len;
}

static size_t upload_data_cb(void *ptr, size_t size, size_t nmemb,
			     void *user_data)
{
	struct upload_buffer *ub = (struct upload_buffer *)user_data;
	unsigned int len = size * nmemb;

	if (len > ub->len - ub->pos)
		len = ub->len - ub->pos;

	if (len) {
		memcpy(ptr, (char*)ub->buf + ub->pos, len);
		ub->pos += len;
	}

	return len;
}

#if LIBCURL_VERSION_NUM >= 0x071200
static int seek_data_cb(void *user_data, curl_off_t offset, int origin)
{
	struct upload_buffer *ub = (struct upload_buffer *)user_data;
	
	switch (origin) {
	case SEEK_SET:
		ub->pos = (size_t)offset;
		break;
	case SEEK_CUR:
		ub->pos += (size_t)offset;
		break;
	case SEEK_END:
		ub->pos = ub->len + (size_t)offset;
		break;
	default:
		return 1; /* CURL_SEEKFUNC_FAIL */
	}

	return 0; /* CURL_SEEKFUNC_OK */
}
#endif

static size_t resp_hdr_cb(void *ptr, size_t size, size_t nmemb, void *user_data)
{
	struct header_info *hi = (struct header_info *)user_data;
	size_t remlen, slen, ptrlen = size * nmemb;
	char *rem, *val = NULL, *key = NULL;
	void *tmp;

	val = (char*)calloc(1, ptrlen);
	key = (char*)calloc(1, ptrlen);
	if (!key || !val)
		goto out;

	tmp = memchr(ptr, ':', ptrlen);
	if (!tmp || (tmp == ptr))	/* skip empty keys / blanks */
		goto out;
	slen = (size_t)((char*)tmp - (char*)ptr);
	if ((slen + 1) == ptrlen)	/* skip key w/ no value */
		goto out;
	memcpy(key, ptr, slen);		/* store & nul term key */
	key[slen] = 0;

	rem = (char*)ptr + slen + 1;		/* trim value's leading whitespace */
	remlen = ptrlen - slen - 1;
	while ((remlen > 0) && (isspace(*rem))) {
		remlen--;
		rem++;
	}

	memcpy(val, rem, remlen);	/* store value, trim trailing ws */
	val[remlen] = 0;
	while ((*val) && (isspace(val[strlen(val) - 1]))) {
		val[strlen(val) - 1] = 0;
	}
	if (!*val)			/* skip blank value */
		goto out;

	if (!strcasecmp("X-Long-Polling", key)) {
		hi->lp_path = val;	/* steal memory reference */
		val = NULL;
	}

	if (!strcasecmp("X-Reject-Reason", key)) {
		hi->reason = val;	/* steal memory reference */
		val = NULL;
	}

	if (!strcasecmp("X-Stratum", key)) {
		hi->stratum_url = val;	/* steal memory reference */
		val = NULL;
	}

out:
	free(key);
	free(val);
	return ptrlen;
}

#if LIBCURL_VERSION_NUM >= 0x070f06
static int sockopt_keepalive_cb(void *userdata, curl_socket_t fd,
	curlsocktype purpose)
{
	int keepalive = 1;
	int tcp_keepcnt = 3;
	int tcp_keepidle = 50;
	int tcp_keepintvl = 50;

#ifndef WIN32
	if (unlikely(setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive,
		sizeof(keepalive))))
		return 1;
#ifdef __linux
	if (unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPCNT,
		&tcp_keepcnt, sizeof(tcp_keepcnt))))
		return 1;
	if (unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPIDLE,
		&tcp_keepidle, sizeof(tcp_keepidle))))
		return 1;
	if (unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPINTVL,
		&tcp_keepintvl, sizeof(tcp_keepintvl))))
		return 1;
#endif /* __linux */
#ifdef __APPLE_CC__
	if (unlikely(setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE,
		&tcp_keepintvl, sizeof(tcp_keepintvl))))
		return 1;
#endif /* __APPLE_CC__ */
#else /* WIN32 */
	struct tcp_keepalive vals;
	vals.onoff = 1;
	vals.keepalivetime = tcp_keepidle * 1000;
	vals.keepaliveinterval = tcp_keepintvl * 1000;
	DWORD outputBytes;
	if (unlikely(WSAIoctl(fd, SIO_KEEPALIVE_VALS, &vals, sizeof(vals),
		NULL, 0, &outputBytes, NULL, NULL)))
		return 1;
#endif /* WIN32 */

	return 0;
}
#endif

json_t *json_rpc_call(struct pool_params* pool, CURL *curl, const char *url,
		      const char *userpass, const char *rpc_req,
		      bool longpoll_scan, bool longpoll, int *curl_err)
{
	json_t *val, *err_val, *res_val;
	int rc;
	struct data_buffer all_data = {0};
	struct upload_buffer upload_data;
	json_error_t err;
	struct curl_slist *headers = NULL;
	char len_hdr[64];
	char curl_err_str[CURL_ERROR_SIZE];
	long timeout = longpoll ? opt_timeout : 30;
	struct header_info hi = {0};

	/* it is assumed that 'curl' is freshly [re]initialized at this pt */

	if (opt_protocol)
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
	curl_easy_setopt(curl, CURLOPT_URL, url);
	if (opt_cert)
		curl_easy_setopt(curl, CURLOPT_CAINFO, opt_cert);
	curl_easy_setopt(curl, CURLOPT_ENCODING, "");
	curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1);
	curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, all_data_cb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &all_data);
	curl_easy_setopt(curl, CURLOPT_READFUNCTION, upload_data_cb);
	curl_easy_setopt(curl, CURLOPT_READDATA, &upload_data);
#if LIBCURL_VERSION_NUM >= 0x071200
	curl_easy_setopt(curl, CURLOPT_SEEKFUNCTION, &seek_data_cb);
	curl_easy_setopt(curl, CURLOPT_SEEKDATA, &upload_data);
#endif
	curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_err_str);
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
	curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, resp_hdr_cb);
	curl_easy_setopt(curl, CURLOPT_HEADERDATA, &hi);
	if (opt_proxy) {
		curl_easy_setopt(curl, CURLOPT_PROXY, opt_proxy);
		curl_easy_setopt(curl, CURLOPT_PROXYTYPE, opt_proxy_type);
	}
	if (userpass) {
		curl_easy_setopt(curl, CURLOPT_USERPWD, userpass);
		curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
	}
#if LIBCURL_VERSION_NUM >= 0x070f06
	if (longpoll)
		curl_easy_setopt(curl, CURLOPT_SOCKOPTFUNCTION, sockopt_keepalive_cb);
#endif
	curl_easy_setopt(curl, CURLOPT_POST, 1);

	if (opt_protocol)
		applog(LOG_DEBUG, "JSON protocol request:\n%s\n", rpc_req);

	upload_data.buf = rpc_req;
	upload_data.len = strlen(rpc_req);
	upload_data.pos = 0;
	sprintf(len_hdr, "Content-Length: %lu",
		(unsigned long) upload_data.len);

	headers = curl_slist_append(headers, "Content-Type: application/json");
	headers = curl_slist_append(headers, len_hdr);
	headers = curl_slist_append(headers, "User-Agent: " USER_AGENT);
	headers = curl_slist_append(headers, "X-Mining-Extensions: midstate");
	headers = curl_slist_append(headers, "Accept:"); /* disable Accept hdr*/
	headers = curl_slist_append(headers, "Expect:"); /* disable Expect hdr*/

	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

	rc = curl_easy_perform(curl);
	if (curl_err != NULL)
		*curl_err = rc;
	if (rc) {
		if (!(longpoll && rc == CURLE_OPERATION_TIMEDOUT))
			applog(LOG_ERR, "HTTP request failed: %s", curl_err_str);
		goto err_out;
	}

    /* If X-Stratum was found, activate Stratum */
	if (want_stratum && !pool->have_stratum && !pool->stratum.url && hi.stratum_url &&
	    !strncasecmp(hi.stratum_url, "stratum+tcp://", 14)) {
		pool->have_stratum = true;
        pool->stratum.url = hi.stratum_url;
		hi.stratum_url = NULL;
	}

	/* If X-Long-Polling was found, activate long polling */
	if (longpoll_scan && !pool->have_longpoll && !pool->longpoll_url && hi.lp_path) {
		pool->have_longpoll = true;
        pool->longpoll_url = hi.lp_path;
		hi.lp_path = NULL;
	}

    if (!all_data.buf) {
		applog(LOG_ERR, "Empty data received in json_rpc_call.");
		goto err_out;
	}

	val = JSON_LOADS((const char*)all_data.buf, &err);
	if (!val) {
		applog(LOG_ERR, "JSON decode failed(%d): %s", err.line, err.text);
		goto err_out;
	}

	if (opt_protocol) {
		char *s = json_dumps(val, JSON_INDENT(3));
		applog(LOG_DEBUG, "JSON protocol response:\n%s", s);
		free(s);
	}

	/* JSON-RPC valid response returns a non-null 'result',
	 * and a null 'error'. */
	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");

	if (!res_val || json_is_null(res_val) ||
	    (err_val && !json_is_null(err_val))) {
		char *s;

		if (err_val)
			s = json_dumps(err_val, JSON_INDENT(3));
		else
			s = strdup("(unknown reason)");

		applog(LOG_ERR, "JSON-RPC call failed: %s", s);

		free(s);

		goto err_out;
	}

	if (hi.reason)
		json_object_set_new(val, "reject-reason", json_string(hi.reason));

	databuf_free(&all_data);
	curl_slist_free_all(headers);
	curl_easy_reset(curl);
	return val;

err_out:
	free(hi.lp_path);
	free(hi.reason);
	free(hi.stratum_url);
	databuf_free(&all_data);
	curl_slist_free_all(headers);
	curl_easy_reset(curl);
	return NULL;
}

char *bin2hex(const unsigned char *p, size_t len)
{
	unsigned int i;
	char *s = (char *)malloc((len * 2) + 1);
	if (!s)
		return NULL;

	for (i = 0; i < len; i++)
		sprintf(s + (i * 2), "%02x", (unsigned int) p[i]);

	return s;
}

bool hex2bin(unsigned char *p, const char *hexstr, size_t len)
{
	char hex_byte[3];
	char *ep;

	hex_byte[2] = '\0';

	while (*hexstr && len) {
		if (!hexstr[1]) {
			applog(LOG_ERR, "hex2bin str truncated");
			return false;
		}
		hex_byte[0] = hexstr[0];
		hex_byte[1] = hexstr[1];
		*p = (unsigned char) strtol(hex_byte, &ep, 16);
		if (*ep) {
			applog(LOG_ERR, "hex2bin failed on '%s'", hex_byte);
			return false;
		}
		p++;
		hexstr += 2;
		len--;
	}

	return (len == 0 && *hexstr == 0) ? true : false;
}

/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *x,
	struct timeval *y)
{
	/* Perform the carry for the later subtraction by updating Y. */
	if (x->tv_usec < y->tv_usec) {
		int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
		y->tv_usec -= 1000000 * nsec;
		y->tv_sec += nsec;
	}
	if (x->tv_usec - y->tv_usec > 1000000) {
		int nsec = (x->tv_usec - y->tv_usec) / 1000000;
		y->tv_usec += 1000000 * nsec;
		y->tv_sec -= nsec;
	}

	/* Compute the time remaining to wait.
	 * `tv_usec' is certainly positive. */
	result->tv_sec = x->tv_sec - y->tv_sec;
	result->tv_usec = x->tv_usec - y->tv_usec;

	/* Return 1 if result is negative. */
	return x->tv_sec < y->tv_sec;
}

bool fulltest(const uint32_t *hash, const uint32_t *target)
{
	int i;
	bool rc = true;
	
	for (i = 7; i >= 0; i--) {
		if (hash[i] > target[i]) {
			rc = false;
			break;
		}
		if (hash[i] < target[i]) {
			rc = true;
			break;
		}
	}

	if (opt_debug) {
		uint32_t hash_be[8], target_be[8];
		char *hash_str, *target_str;
		
		for (i = 0; i < 8; i++) {
			be32enc(hash_be + i, hash[7 - i]);
			be32enc(target_be + i, target[7 - i]);
		}
		hash_str = bin2hex((unsigned char *)hash_be, 32);
		target_str = bin2hex((unsigned char *)target_be, 32);

		applog(LOG_DEBUG, "DEBUG: %s\nHash:   %s\nTarget: %s",
			rc ? "hash <= target"
			   : "hash > target (false positive)",
			hash_str,
			target_str);

		free(hash_str);
		free(target_str);
	}

	return rc;
}

void diff_to_target(uint32_t *target, double diff)
{
	uint64_t m;
	int k;
	
	for (k = 6; k > 0 && diff > 1.0; k--)
		diff /= 4294967296.0;
	m = (uint64_t)(4294901760.0 / diff);
	if (m == 0 && k == 6)
		memset(target, 0xff, 32);
	else {
		memset(target, 0, 32);
		target[k] = (uint32_t)m;
		target[k + 1] = (uint32_t)(m >> 32);
	}
}

#ifdef WIN32
#define socket_blocks() (WSAGetLastError() == WSAEWOULDBLOCK)
#else
#define socket_blocks() (errno == EAGAIN || errno == EWOULDBLOCK)
#endif

static bool send_line(curl_socket_t sock, char *s)
{
	ssize_t len, sent = 0;
	
	len = strlen(s);
	s[len++] = '\n';

	while (len > 0) {
		struct timeval timeout = {15, 0};
		ssize_t n;
		fd_set wd;

		FD_ZERO(&wd);
		FD_SET(sock, &wd);
		if (select(sock + 1, NULL, &wd, NULL, &timeout) < 1)
			return false;
		n = send(sock, s + sent, len, 0);
		if (n < 0) {
			if (!socket_blocks())
				return false;
			n = 0;
		}
		sent += n;
		len -= n;
	}

	return true;
}

bool stratum_send_line(struct stratum_ctx *sctx, char *s)
{
	bool ret = false;

	if (opt_protocol)
		applog(LOG_DEBUG, "> %s", s);

	pthread_mutex_lock(&sctx->sock_lock);
	ret = send_line(sctx->sock, s);
	pthread_mutex_unlock(&sctx->sock_lock);

	return ret;
}

static bool socket_full(curl_socket_t sock, int timeout)
{
	struct timeval tv;
	fd_set rd;

	// CB: do not block on the socket for more than 1 second
	for (int t = 0; t < timeout && !abort_flag; ++t)
	{
		FD_ZERO(&rd);
		FD_SET(sock, &rd);
		tv.tv_sec = 1;
		tv.tv_usec = 0;
		if (select(sock + 1, &rd, NULL, NULL, &tv) > 0)
			return true;
	}
	return false;
}

bool stratum_socket_full(struct stratum_ctx *sctx, int timeout)
{
	return strlen(sctx->sockbuf) || socket_full(sctx->sock, timeout);
}

#define RBUFSIZE 2048
#define RECVSIZE (RBUFSIZE - 4)

static void stratum_buffer_append(struct stratum_ctx *sctx, const char *s)
{
	size_t old, snew;

	old = strlen(sctx->sockbuf);
	snew = old + strlen(s) + 1;
	if (snew >= sctx->sockbuf_size) {
		sctx->sockbuf_size = snew + (RBUFSIZE - (snew % RBUFSIZE));
		sctx->sockbuf = (char*)realloc(sctx->sockbuf, sctx->sockbuf_size);
	}
	strcpy(sctx->sockbuf + old, s);
}

char *stratum_recv_line(struct stratum_ctx *sctx)
{
	ssize_t len, buflen;
	char *tok, *sret = NULL;

	if (!strchr(sctx->sockbuf, '\n')) {
		bool ret = true;
		time_t rstart;

		time(&rstart);
		if (!socket_full(sctx->sock, 60)) {
			applog(LOG_ERR, "stratum_recv_line timed out");
			goto out;
		}
		do {
			char s[RBUFSIZE];
			ssize_t n;

			memset(s, 0, RBUFSIZE);
			n = recv(sctx->sock, s, RECVSIZE, 0);
			if (!n) {
				ret = false;
				break;
			}
			if (n < 0) {
				if (!socket_blocks() || !socket_full(sctx->sock, 1)) {
					ret = false;
					break;
				}
			} else
				stratum_buffer_append(sctx, s);
		} while (time(NULL) - rstart < 60 && !strchr(sctx->sockbuf, '\n'));

		if (!ret) {
			applog(LOG_ERR, "stratum_recv_line failed");
			goto out;
		}
	}

	buflen = strlen(sctx->sockbuf);
	tok = strchr(sctx->sockbuf, '\n');
	if (!tok) {
		applog(LOG_ERR, "stratum_recv_line failed to parse a newline-terminated string (length %d)", buflen);
		sctx->sockbuf[0] = '\0';
		goto out;
	} else *tok = '\0';
	sret = strdup(sctx->sockbuf);
	len = strlen(sret);
	if (buflen > len + 1)
		memmove(sctx->sockbuf, sctx->sockbuf + len + 1, buflen - len);
	else
		sctx->sockbuf[0] = '\0';

out:
	if (sret && opt_protocol)
		applog(LOG_DEBUG, "< %s", sret);
	return sret;
}

#if LIBCURL_VERSION_NUM >= 0x071101
static curl_socket_t opensocket_grab_cb(void *clientp, curlsocktype purpose,
	struct curl_sockaddr *addr)
{
	curl_socket_t *sock = (curl_socket_t *)clientp;
	*sock = socket(addr->family, addr->socktype, addr->protocol);
	return *sock;
}
#endif

bool stratum_connect(struct stratum_ctx *sctx, const char *url)
{
	CURL *curl;
	int rc;

	pthread_mutex_lock(&sctx->sock_lock);
	if (sctx->curl)
		curl_easy_cleanup(sctx->curl);
	sctx->curl = curl_easy_init();
	if (!sctx->curl) {
		applog(LOG_ERR, "CURL initialization failed");
		pthread_mutex_unlock(&sctx->sock_lock);
		return false;
	}
	curl = sctx->curl;
	if (!sctx->sockbuf) {
		sctx->sockbuf = (char*)calloc(RBUFSIZE, 1);
		sctx->sockbuf_size = RBUFSIZE;
	}
	sctx->sockbuf[0] = '\0';
	pthread_mutex_unlock(&sctx->sock_lock);

	if (url != sctx->url) {
		free(sctx->url);
		sctx->url = strdup(url);
	}
	free(sctx->curl_url);
	sctx->curl_url = (char*)malloc(strlen(url));
	sprintf(sctx->curl_url, "http%s", strstr(url, "://"));

	if (opt_protocol)
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
	curl_easy_setopt(curl, CURLOPT_URL, sctx->curl_url);
	curl_easy_setopt(curl, CURLOPT_FRESH_CONNECT, 1);
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30);
	curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, sctx->curl_err_str);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1);
	curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1);
	if (opt_proxy && opt_proxy_type != CURLPROXY_HTTP) {
		curl_easy_setopt(curl, CURLOPT_PROXY, opt_proxy);
		curl_easy_setopt(curl, CURLOPT_PROXYTYPE, opt_proxy_type);
	} else if (getenv("http_proxy")) {
		if (getenv("all_proxy"))
			curl_easy_setopt(curl, CURLOPT_PROXY, getenv("all_proxy"));
		else if (getenv("ALL_PROXY"))
			curl_easy_setopt(curl, CURLOPT_PROXY, getenv("ALL_PROXY"));
		else
			curl_easy_setopt(curl, CURLOPT_PROXY, "");
	}
#if LIBCURL_VERSION_NUM >= 0x070f06
	curl_easy_setopt(curl, CURLOPT_SOCKOPTFUNCTION, sockopt_keepalive_cb);
#endif
#if LIBCURL_VERSION_NUM >= 0x071101
	curl_easy_setopt(curl, CURLOPT_OPENSOCKETFUNCTION, opensocket_grab_cb);
	curl_easy_setopt(curl, CURLOPT_OPENSOCKETDATA, &sctx->sock);
#endif
	curl_easy_setopt(curl, CURLOPT_CONNECT_ONLY, 1);

	rc = curl_easy_perform(curl);
	if (rc) {
		applog(LOG_ERR, "Stratum connection failed: %s", sctx->curl_err_str);
		curl_easy_cleanup(curl);
		sctx->curl = NULL;
		return false;
	}

#if LIBCURL_VERSION_NUM < 0x071101
	/* CURLINFO_LASTSOCKET is broken on Win64; only use it as a last resort */
	curl_easy_getinfo(curl, CURLINFO_LASTSOCKET, (long *)&sctx->sock);
#endif

	return true;
}

void stratum_disconnect(struct stratum_ctx *sctx)
{
	pthread_mutex_lock(&sctx->sock_lock);
	if (sctx->curl) {
		curl_easy_cleanup(sctx->curl);
		sctx->curl = NULL;
		sctx->sockbuf[0] = '\0';
	}
	pthread_mutex_unlock(&sctx->sock_lock);
}

static const char *get_stratum_session_id(json_t *val)
{
	json_t *arr_val;
	int i, n;

	arr_val = json_array_get(val, 0);
	if (!arr_val || !json_is_array(arr_val))
		return NULL;
	n = json_array_size(arr_val);
	for (i = 0; i < n; i++) {
		const char *notify;
		json_t *arr = json_array_get(arr_val, i);

		if (!arr || !json_is_array(arr))
			break;
		notify = json_string_value(json_array_get(arr, 0));
		if (!notify)
			continue;
		if (!strcasecmp(notify, "mining.notify"))
			return json_string_value(json_array_get(arr, 1));
	}
	return NULL;
}

bool stratum_subscribe(struct stratum_ctx *sctx)
{
	char *s, *sret = NULL;
	const char *sid, *xnonce1;
	int xn2_size;
	json_t *val = NULL, *res_val, *err_val;
	json_error_t err;
	bool ret = false, retry = false;

start:
	s = (char*)malloc(128 + (sctx->session_id ? strlen(sctx->session_id) : 0));
	if (retry)
		sprintf(s, "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": []}");
	else if (sctx->session_id)
		sprintf(s, "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": [\"" USER_AGENT "\", \"%s\"]}", sctx->session_id);
	else
		sprintf(s, "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": [\"" USER_AGENT "\"]}");

	if (!stratum_send_line(sctx, s))
		goto out;

	if (!socket_full(sctx->sock, 30)) {
		applog(LOG_ERR, "stratum_subscribe timed out");
		goto out;
	}

	sret = stratum_recv_line(sctx);
	if (!sret)
		goto out;

	val = JSON_LOADS(sret, &err);
	free(sret);
	if (!val) {
		applog(LOG_ERR, "JSON decode failed(%d): %s", err.line, err.text);
		goto out;
	}

	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");

	if (!res_val || json_is_null(res_val) ||
	    (err_val && !json_is_null(err_val))) {
		if (opt_debug || retry) {
			free(s);
			if (err_val)
				s = json_dumps(err_val, JSON_INDENT(3));
			else
				s = strdup("(unknown reason)");
			applog(LOG_ERR, "JSON-RPC call failed: %s", s);
		}
		goto out;
	}

	sid = get_stratum_session_id(res_val);
	if (opt_debug && !sid)
		applog(LOG_DEBUG, "Failed to get Stratum session id");
	xnonce1 = json_string_value(json_array_get(res_val, 1));
	if (!xnonce1) {
		applog(LOG_ERR, "Failed to get extranonce1");
		goto out;
	}
	xn2_size = json_integer_value(json_array_get(res_val, 2));
	if (!xn2_size) {
		applog(LOG_ERR, "Failed to get extranonce2_size");
		goto out;
	}

	pthread_mutex_lock(&sctx->work_lock);
	free(sctx->session_id);
	free(sctx->xnonce1);
	sctx->session_id = sid ? strdup(sid) : NULL;
	sctx->xnonce1_size = strlen(xnonce1) / 2;
	sctx->xnonce1 = (unsigned char*)malloc(sctx->xnonce1_size);
	hex2bin(sctx->xnonce1, xnonce1, sctx->xnonce1_size);
	sctx->xnonce2_size = xn2_size;
	sctx->next_diff = 1.0;
	pthread_mutex_unlock(&sctx->work_lock);

	if (opt_debug && sid)
		applog(LOG_DEBUG, "Stratum session id: %s", sctx->session_id);

	ret = true;

out:
	free(s);
	if (val)
		json_decref(val);

	if (!ret) {
		if (sret && !retry) {
			retry = true;
			goto start;
		}
	}

	return ret;
}

bool stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass)
{
	json_t *val = NULL, *res_val, *err_val;
	char *s, *sret;
	json_error_t err;
	bool ret = false;

	s = (char*)malloc(80 + strlen(user) + strlen(pass));
	sprintf(s, "{\"id\": 2, \"method\": \"mining.authorize\", \"params\": [\"%s\", \"%s\"]}",
	        user, pass);

	if (!stratum_send_line(sctx, s))
		goto out;

	while (1) {
		sret = stratum_recv_line(sctx);
		if (!sret)
			goto out;
		if (!stratum_handle_method(sctx, sret))
			break;
		free(sret);
	}

	val = JSON_LOADS(sret, &err);
	free(sret);
	if (!val) {
		applog(LOG_ERR, "JSON decode failed(%d): %s", err.line, err.text);
		goto out;
	}

	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");

	if (!res_val || json_is_false(res_val) ||
	    (err_val && !json_is_null(err_val)))  {
		applog(LOG_ERR, "Stratum authentication failed");
		goto out;
	}

	ret = true;

out:
	free(s);
	if (val)
		json_decref(val);

	return ret;
}

static bool stratum_notify(struct stratum_ctx *sctx, json_t *params)
{
	const char *job_id, *prevhash, *coinb1, *coinb2, *version, *nbits, *ntime;
	size_t coinb1_size, coinb2_size;
	bool clean, ret = false;
	int merkle_count, i;
	json_t *merkle_arr;
	unsigned char **merkle;

	job_id = json_string_value(json_array_get(params, 0));
	prevhash = json_string_value(json_array_get(params, 1));
	coinb1 = json_string_value(json_array_get(params, 2));
	coinb2 = json_string_value(json_array_get(params, 3));
	merkle_arr = json_array_get(params, 4);
	if (!merkle_arr || !json_is_array(merkle_arr))
		goto out;
	merkle_count = json_array_size(merkle_arr);
	version = json_string_value(json_array_get(params, 5));
	nbits = json_string_value(json_array_get(params, 6));
	ntime = json_string_value(json_array_get(params, 7));
	clean = json_is_true(json_array_get(params, 8));

	if (!job_id || !prevhash || !coinb1 || !coinb2 || !version || !nbits || !ntime ||
	    strlen(prevhash) != 64 || strlen(version) != 8 ||
	    strlen(nbits) != 8 || strlen(ntime) != 8) {
		applog(LOG_ERR, "Stratum notify: invalid parameters");
		goto out;
	}
	merkle = (unsigned char**)malloc(merkle_count * sizeof(char *));
	for (i = 0; i < merkle_count; i++) {
		const char *s = json_string_value(json_array_get(merkle_arr, i));
		if (!s || strlen(s) != 64) {
			while (i--)
				free(merkle[i]);
			free(merkle);
			applog(LOG_ERR, "Stratum notify: invalid Merkle branch");
			goto out;
		}
		merkle[i] = (unsigned char*)malloc(32);
		hex2bin(merkle[i], s, 32);
	}

	pthread_mutex_lock(&sctx->work_lock);

	coinb1_size = strlen(coinb1) / 2;
	coinb2_size = strlen(coinb2) / 2;
	sctx->job.coinbase_size = coinb1_size + sctx->xnonce1_size +
	                          sctx->xnonce2_size + coinb2_size;
	sctx->job.coinbase = (unsigned char*)realloc(sctx->job.coinbase, sctx->job.coinbase_size);
	sctx->job.xnonce2 = sctx->job.coinbase + coinb1_size + sctx->xnonce1_size;
	hex2bin(sctx->job.coinbase, coinb1, coinb1_size);
	memcpy(sctx->job.coinbase + coinb1_size, sctx->xnonce1, sctx->xnonce1_size);
	if (!sctx->job.job_id || strcmp(sctx->job.job_id, job_id))
		memset(sctx->job.xnonce2, 0, sctx->xnonce2_size);
	hex2bin(sctx->job.xnonce2 + sctx->xnonce2_size, coinb2, coinb2_size);

	free(sctx->job.job_id);
	sctx->job.job_id = strdup(job_id);
	hex2bin(sctx->job.prevhash, prevhash, 32);

	for (i = 0; i < sctx->job.merkle_count; i++)
		free(sctx->job.merkle[i]);
	free(sctx->job.merkle);
	sctx->job.merkle = merkle;
	sctx->job.merkle_count = merkle_count;

	hex2bin(sctx->job.version, version, 4);
	hex2bin(sctx->job.nbits, nbits, 4);
	hex2bin(sctx->job.ntime, ntime, 4);
	sctx->job.clean = clean;

	sctx->job.diff = sctx->next_diff;

	pthread_mutex_unlock(&sctx->work_lock);

	ret = true;

out:
	return ret;
}

static bool stratum_set_difficulty(struct stratum_ctx *sctx, json_t *params)
{
	double diff;

	diff = json_number_value(json_array_get(params, 0));
	if (diff == 0)
		return false;

	pthread_mutex_lock(&sctx->work_lock);
	sctx->next_diff = diff;
	pthread_mutex_unlock(&sctx->work_lock);

	if (opt_debug)
		applog(LOG_DEBUG, "Stratum difficulty set to %g", diff);

	return true;
}

static bool stratum_reconnect(struct stratum_ctx *sctx, json_t *params)
{
	json_t *port_val;
	const char *host;
	int port;

	host = json_string_value(json_array_get(params, 0));
	port_val = json_array_get(params, 1);
	if (json_is_string(port_val))
		port = atoi(json_string_value(port_val));
	else
		port = json_integer_value(port_val);
	if (!host || !port)
		return false;
	
	free(sctx->url);
	sctx->url = (char*)malloc(32 + strlen(host));
	sprintf(sctx->url, "stratum+tcp://%s:%d", host, port);

	applog(LOG_NOTICE, "Server requested reconnection to %s", sctx->url);

	stratum_disconnect(sctx);

	return true;
}

static bool stratum_get_version(struct stratum_ctx *sctx, json_t *id)
{
	char *s;
	json_t *val;
	bool ret;
	
	if (!id || json_is_null(id))
		return false;

	val = json_object();
	json_object_set(val, "id", id);
	json_object_set_new(val, "error", json_null());
	json_object_set_new(val, "result", json_string(USER_AGENT));
	s = json_dumps(val, 0);
	ret = stratum_send_line(sctx, s);
	json_decref(val);
	free(s);

	return ret;
}

static bool stratum_show_message(struct stratum_ctx *sctx, json_t *id, json_t *params)
{
	char *s;
	json_t *val;
	bool ret;

	val = json_array_get(params, 0);
	if (val)
		applog(LOG_NOTICE, "MESSAGE FROM SERVER: %s", json_string_value(val));
	
	if (!id || json_is_null(id))
		return true;

	val = json_object();
	json_object_set(val, "id", id);
	json_object_set_new(val, "error", json_null());
	json_object_set_new(val, "result", json_true());
	s = json_dumps(val, 0);
	ret = stratum_send_line(sctx, s);
	json_decref(val);
	free(s);

	return ret;
}

bool stratum_handle_method(struct stratum_ctx *sctx, const char *s)
{
	json_t *val, *id, *params;
	json_error_t err;
	const char *method;
	bool ret = false;

	val = JSON_LOADS(s, &err);
	if (!val) {
		applog(LOG_ERR, "JSON decode failed(%d): %s", err.line, err.text);
		goto out;
	}

	method = json_string_value(json_object_get(val, "method"));
	if (!method)
		goto out;
	id = json_object_get(val, "id");
	params = json_object_get(val, "params");

	if (!strcasecmp(method, "mining.notify")) {
		ret = stratum_notify(sctx, params);
		goto out;
	}
	if (!strcasecmp(method, "mining.set_difficulty")) {
		ret = stratum_set_difficulty(sctx, params);
		goto out;
	}
	if (!strcasecmp(method, "client.reconnect")) {
		ret = stratum_reconnect(sctx, params);
		goto out;
	}
	if (!strcasecmp(method, "client.get_version")) {
		ret = stratum_get_version(sctx, id);
		goto out;
	}
	if (!strcasecmp(method, "client.show_message")) {
		ret = stratum_show_message(sctx, id, params);
		goto out;
	}

out:
	if (val)
		json_decref(val);

	return ret;
}

struct thread_q *tq_new(void)
{
	struct thread_q *tq;

	tq = (struct thread_q *)calloc(1, sizeof(*tq));
	if (!tq)
		return NULL;

	INIT_LIST_HEAD(&tq->q);
	pthread_mutex_init(&tq->mutex, NULL);
	pthread_cond_init(&tq->cond, NULL);

	return tq;
}

void tq_free(struct thread_q *tq)
{
	struct tq_ent *ent, *iter;

	if (!tq)
		return;

	list_for_each_entry_safe(ent, iter, &tq->q, q_node, struct tq_ent, struct tq_ent) {
		list_del(&ent->q_node);
		free(ent);
	}

	pthread_cond_destroy(&tq->cond);
	pthread_mutex_destroy(&tq->mutex);

	memset(tq, 0, sizeof(*tq));	/* poison */
	free(tq);
}

static void tq_freezethaw(struct thread_q *tq, bool frozen)
{
	pthread_mutex_lock(&tq->mutex);

	tq->frozen = frozen;

	pthread_cond_signal(&tq->cond);
	pthread_mutex_unlock(&tq->mutex);
}

void tq_freeze(struct thread_q *tq)
{
	tq_freezethaw(tq, true);
}

void tq_thaw(struct thread_q *tq)
{
	tq_freezethaw(tq, false);
}

bool tq_push(struct thread_q *tq, void *data)
{
	struct tq_ent *ent;
	bool rc = true;

	ent = (struct tq_ent *)calloc(1, sizeof(*ent));
	if (!ent)
		return false;

	ent->data = data;
	INIT_LIST_HEAD(&ent->q_node);

	pthread_mutex_lock(&tq->mutex);

	if (!tq->frozen) {
		list_add_tail(&ent->q_node, &tq->q);
	} else {
		free(ent);
		rc = false;
	}

	pthread_cond_signal(&tq->cond);
	pthread_mutex_unlock(&tq->mutex);

	return rc;
}

void *tq_pop(struct thread_q *tq, const struct timespec *abstime)
{
	struct tq_ent *ent;
	void *rval = NULL;
	int rc;

	pthread_mutex_lock(&tq->mutex);

	if (!list_empty(&tq->q))
		goto pop;

	if (abstime)
		rc = pthread_cond_timedwait(&tq->cond, &tq->mutex, abstime);
	else
		rc = pthread_cond_wait(&tq->cond, &tq->mutex);
	if (rc)
		goto out;
	if (list_empty(&tq->q))
		goto out;

pop:
	ent = list_entry(tq->q.next, struct tq_ent, q_node);
	rval = ent->data;

	list_del(&ent->q_node);
	free(ent);

out:
	pthread_mutex_unlock(&tq->mutex);
	return rval;
}
