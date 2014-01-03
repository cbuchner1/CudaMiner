
typedef enum cpu_flags_x86_t { }cpu_flags_x86;

typedef enum cpu_vendors_x86_t {
	cpu_nobody,
	cpu_intel,
	cpu_amd
} cpu_vendors_x86;

typedef struct x86_regs_t {
	uint32_t eax, ebx, ecx, edx;
} x86_regs;


#if defined(SCRYPT_TEST_SPEED)
size_t cpu_detect_mask = (size_t)-1;
#endif

static size_t
detect_cpu(void) {
	size_t cpu_flags = 0;
	return cpu_flags;
}

#if defined(SCRYPT_TEST_SPEED)
static const char *
get_top_cpuflag_desc(size_t flag) {
	return "Basic";
}
#endif

#define asm_calling_convention
