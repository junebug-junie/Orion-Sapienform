/* Authored ABI test double, NOT an NES/SNES emulator or a ROM. */
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
struct system_info { const char *name, *version, *extensions; bool fullpath, block; };
struct geometry { unsigned width, height, max_width, max_height; float aspect; };
struct timing { double fps, sample_rate; };
struct av_info { struct geometry geometry; struct timing timing; };
struct game { const char *path; const void *data; size_t size; const char *meta; };
static bool (*environment)(unsigned, void *);
static void (*video)(const void *, unsigned, unsigned, size_t);
static void (*audio)(int16_t, int16_t);
static size_t (*batch)(const int16_t *, size_t);
static void (*poll)(void);
static int16_t (*input)(unsigned, unsigned, unsigned, unsigned);
static uint32_t state;
unsigned retro_api_version(void) { return 1; }
void retro_set_environment(bool (*f)(unsigned, void *)) { environment = f; }
void retro_set_video_refresh(void (*f)(const void *, unsigned, unsigned, size_t)) { video = f; }
void retro_set_audio_sample(void (*f)(int16_t, int16_t)) { audio = f; }
void retro_set_audio_sample_batch(size_t (*f)(const int16_t *, size_t)) { batch = f; }
void retro_set_input_poll(void (*f)(void)) { poll = f; }
void retro_set_input_state(int16_t (*f)(unsigned, unsigned, unsigned, unsigned)) { input = f; }
void retro_init(void) { state = 0; }
void retro_deinit(void) {}
void retro_get_system_info(struct system_info *info) {
    *info = (struct system_info){"fixture", "1", "nes", false, false};
}
void retro_get_system_av_info(struct av_info *info) {
    *info = (struct av_info){{2, 2, 2, 2, 1}, {60, 48000}};
}
bool retro_load_game(const struct game *game) {
    int format = 1;
    return game->size == 7 && memcmp(game->data, "fixture", 7) == 0 && environment(10, &format);
}
void retro_unload_game(void) {}
void retro_set_controller_port_device(unsigned port, unsigned device) {}
void retro_run(void) {
    poll();
    state += 1 + input(0, 1, 0, 8) + 10 * input(1, 1, 0, 0);
    uint32_t pixels[4] = {state, state, state, state};
    int16_t samples[4] = {1, 2, 3, 4};
    video(pixels, 2, 2, 8);
    audio(1, 2);
    batch(samples, 2);
}
size_t retro_serialize_size(void) { return sizeof(state); }
bool retro_serialize(void *out, size_t size) {
    if (size != sizeof(state)) return false;
    memcpy(out, &state, size); return true;
}
bool retro_unserialize(const void *in, size_t size) {
    if (size != sizeof(state)) return false;
    memcpy(&state, in, size); return true;
}
