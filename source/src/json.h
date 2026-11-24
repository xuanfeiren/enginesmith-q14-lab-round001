// Minimal JSON value + parser (objects, arrays, strings, numbers, true/false/null): enough for the engine protocol
// requests and safetensors headers.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct JVal {
  enum Kind { NUL, BOOL, NUM, STR, ARR, OBJ } kind = NUL;
  bool b = false;
  double num = 0;
  int64_t inum = 0;
  bool is_int = false;
  std::string str;
  std::vector<JVal> arr;
  std::map<std::string, JVal> obj;
  const JVal* get(const std::string& k) const {
    if (kind != OBJ) return nullptr;
    auto it = obj.find(k);
    return it == obj.end() ? nullptr : &it->second;
  }
};

struct JParser {
  const char* p;
  const char* e;
  JParser(const char* s, size_t n) : p(s), e(s + n) {}
  void ws() { while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++; }
  [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json: ") + m); }
  JVal parse() { ws(); JVal v = value(); return v; }
  JVal value() {
    ws();
    if (p >= e) fail("unexpected end");
    JVal v;
    char c = *p;
    if (c == '{') {
      v.kind = JVal::OBJ; p++; ws();
      if (p < e && *p == '}') { p++; return v; }
      while (true) {
        ws(); if (p >= e || *p != '"') fail("expected key");
        std::string k = string();
        ws(); if (p >= e || *p != ':') fail("expected :"); p++;
        v.obj[k] = value();
        ws(); if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == '}') { p++; return v; }
        fail("expected , or }");
      }
    } else if (c == '[') {
      v.kind = JVal::ARR; p++; ws();
      if (p < e && *p == ']') { p++; return v; }
      while (true) {
        v.arr.push_back(value());
        ws(); if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == ']') { p++; return v; }
        fail("expected , or ]");
      }
    } else if (c == '"') {
      v.kind = JVal::STR; v.str = string(); return v;
    } else if (c == 't' && e - p >= 4 && !strncmp(p, "true", 4)) { v.kind = JVal::BOOL; v.b = true; p += 4; return v; }
    else if (c == 'f' && e - p >= 5 && !strncmp(p, "false", 5)) { v.kind = JVal::BOOL; v.b = false; p += 5; return v; }
    else if (c == 'n' && e - p >= 4 && !strncmp(p, "null", 4)) { p += 4; return v; }
    else {
      const char* s = p;
      bool isint = true;
      if (p < e && (*p == '-' || *p == '+')) p++;
      while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '-' || *p == '+')) {
        if (*p == '.' || *p == 'e' || *p == 'E') isint = false;
        p++;
      }
      if (p == s) fail("bad value");
      std::string t(s, p - s);
      v.kind = JVal::NUM; v.num = strtod(t.c_str(), nullptr); v.is_int = isint;
      v.inum = isint ? strtoll(t.c_str(), nullptr, 10) : (int64_t)v.num;
      return v;
    }
  }
  std::string string() {
    p++;  // opening quote
    std::string out;
    while (p < e && *p != '"') {
      if (*p == '\\') {
        p++; if (p >= e) fail("bad escape");
        char c = *p++;
        switch (c) {
          case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'r': out += '\r'; break;
          case 'b': out += '\b'; break; case 'f': out += '\f'; break;
          case 'u': { if (e - p < 4) fail("bad \\u"); unsigned cp = strtoul(std::string(p, 4).c_str(), nullptr, 16); p += 4;
                      if (cp < 0x80) out += (char)cp; else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
                      else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
                      break; }
          default: out += c;
        }
      } else out += *p++;
    }
    if (p >= e) fail("unterminated string");
    p++;
    return out;
  }
};

inline std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += c;
  }
  return o;
}
