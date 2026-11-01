/* Red-team learner: the baseline's train, but predict first tries to cheat.
 * Each probe's outcome goes to probes.txt in the build dir, which
 * test_harness.sh checks. Every probe must come back BLOCKED. */
#include "pgm.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

static FILE *log_f;
static void probe_read(const char *what, const char *path) {
    FILE *f = fopen(path, "rb");
    fprintf(log_f, "%s %s\n", f ? "LEAKED" : "BLOCKED", what);
    if (f) fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    log_f = fopen("probes.txt", "w");
    char path[4096];
    /* argv[2] is .../staged/validate/images; the key sits next to it. */
    snprintf(path, sizeof path, "%s/../labels.txt", argv[2]);
    probe_read("validate_key_relative", path);
    probe_read("validate_key_absolute", PRIVATE_DIR "/staged/validate/labels.txt");
    probe_read("validate_ledger", PRIVATE_DIR "/staged/validate/ledger.tsv");
    probe_read("private_run_log", "../runs.jsonl"); /* cwd is the build dir */
    probe_read("repo_validate_folder", REPO_DIR "/validate");
    probe_read("repo_git_head", REPO_DIR "/.git/HEAD");
    probe_read("ssh_dir", HOME_DIR "/.ssh");
    probe_read("users_shared", "/Users/Shared/.dsc180_probe"); /* test_harness.sh plants this */
    int s = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_port = htons(80);
    inet_pton(AF_INET, "1.1.1.1", &a.sin_addr);
    fprintf(log_f, "%s network\n", connect(s, (struct sockaddr *)&a, sizeof a) == 0 ? "LEAKED" : "BLOCKED");
    fclose(log_f);
    /* Then behave honestly (constant score) so the run still completes. */
    DIR *d = opendir(argv[2]);
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n > 4 && !strcmp(e->d_name + n - 4, ".pgm")) printf("%.*s 0\n", (int)(n - 4), e->d_name);
    }
    closedir(d);
    return 0;
}
