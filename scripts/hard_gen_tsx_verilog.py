"""Synthesize "hard" training samples for a byte-level source-language classifier.

Targets two known failure clusters of a small Magika-distilled model that sees
the first 4096 bytes tokenized into case-folded words, punctuation runs, and
brackets:

  * TSX React components dominated by JSX markup get misread as XML, so
    synth_hard_tsx() emits markup-heavy .tsx sources with only a thin layer of
    TypeScript (an import, a typed props interface, a small hook call).
  * SystemVerilog with object-oriented testbench code gets misread as Dart or
    C++, so synth_hard_verilog() emits class/typedef/interface-heavy .sv text
    plus classic synthesizable module blocks with always_ff and <= assigns.

Both generators draw every random decision from the caller-supplied
random.Random instance and return 200 B .. ~4 KB of pure-ASCII source bytes.
"""

# --------------------------------------------------------------------------
# word pools
# --------------------------------------------------------------------------

_UI_WORDS = [
    "hero", "panel", "card", "grid", "order", "user", "profile", "banner",
    "sidebar", "modal", "badge", "avatar", "invoice", "ticket", "product",
    "cart", "menu", "footer", "nav", "search", "filter", "result",
    "session", "account", "report", "chart", "widget", "notice", "alert",
    "detail", "summary", "signup", "login", "settings", "theme", "layout",
    "preview", "gallery", "comment", "review", "wizard", "stepper",
]

_HW_WORDS = [
    "packet", "fifo", "uart", "axi", "apb", "dma", "cache", "fetch",
    "decode", "issue", "retire", "flush", "burst", "beat", "lane",
    "frame", "header", "payload", "parity", "crc", "timer", "counter",
    "arbiter", "bridge", "router", "buffer", "queue", "stage", "core",
    "link", "spi", "ctrl", "status", "irq", "sync", "grant", "credit",
    "phase", "scoreboard", "monitor", "driver", "agent", "sequencer",
]

_FILLER_WORDS = [
    "latest", "pending", "active", "archived", "primary", "secondary",
    "remote", "local", "shared", "global", "compact", "expanded",
    "verified", "draft", "stable", "legacy", "custom", "default",
]


def _cap(w):
    return w[0].upper() + w[1:]


def _pascal(rng, pool, n=2):
    return "".join(_cap(w) for w in rng.sample(pool, n))


def _camel(rng, pool, n=2):
    ws = rng.sample(pool, n)
    return ws[0] + "".join(_cap(w) for w in ws[1:])


def _kebab(rng, pool, n=2):
    return "-".join(rng.sample(pool, n))


def _snake(rng, pool, n=2):
    return "_".join(rng.sample(pool, n))


def _phrase(rng, n):
    return " ".join(rng.choice(_FILLER_WORDS + _UI_WORDS) for _ in range(n))


# --------------------------------------------------------------------------
# TSX sub-styles
# --------------------------------------------------------------------------

def _tsx_page(rng):
    """Sub-style 1: nested page-layout component (header/nav/sections/footer)."""
    comp = _pascal(rng, _UI_WORDS)
    base = _kebab(rng, _UI_WORDS)
    title_prop = rng.choice(["title", "heading", "caption", "label"])
    flag_prop = rng.choice(["compact", "dark", "wide", "sticky", "inverted"])
    state = rng.choice(["open", "expanded", "visible", "pinned"])
    setter = "set" + _cap(state)

    L = []
    L.append('import React, { useState } from "react";')
    L.append("")
    L.append("interface %sProps {" % comp)
    L.append("  %s: string;" % title_prop)
    L.append("  %s?: boolean;" % flag_prop)
    L.append("}")
    L.append("")
    L.append("export default function %s({ %s, %s }: %sProps) {" % (comp, title_prop, flag_prop, comp))
    L.append("  const [%s, %s] = useState(%s);" % (state, setter, rng.choice(["false", "true"])))
    L.append("  return (")
    L.append('    <div className="%s">' % base)
    L.append('      <header className="%s__header">' % base)
    L.append('        <h1 className="%s">{%s}</h1>' % (_kebab(rng, _UI_WORDS), title_prop))
    L.append('        <nav className="%s-nav" role="navigation">' % rng.choice(_UI_WORDS))
    for _ in range(rng.randint(2, 4)):
        word = rng.choice(_UI_WORDS)
        L.append('          <a href="/%s" className="nav-link nav-link--%s">%s</a>'
                 % (word, rng.choice(_FILLER_WORDS), _cap(word)))
    L.append("        </nav>")
    L.append('        <button type="button" onClick={() => %s(!%s)}>' % (setter, state))
    L.append("          {%s ? \"Collapse\" : \"Expand\"}" % state)
    L.append("        </button>")
    L.append("      </header>")
    for _ in range(rng.randint(1, 4)):
        sec = _kebab(rng, _UI_WORDS)
        L.append('      <section className="%s" id="%s-%d">' % (sec, rng.choice(_UI_WORDS), rng.randint(1, 99)))
        L.append("        <h2>%s</h2>" % _cap(_phrase(rng, 2)))
        L.append("        <p>%s.</p>" % _cap(_phrase(rng, rng.randint(4, 9))))
        if rng.random() < 0.7:
            img = rng.choice(_UI_WORDS)
            L.append('        <img src="/assets/%s-%d.png" alt="%s" width={%d} height={%d} />'
                     % (img, rng.randint(1, 40), img, rng.choice([320, 480, 640]), rng.choice([120, 180, 240])))
        if rng.random() < 0.5:
            L.append('        <hr className="divider" />')
        L.append("      </section>")
    L.append('      <footer className="%s__footer">' % base)
    L.append("        <span>%s</span>" % _cap(_phrase(rng, 3)))
    L.append('        <small>v{%d}.{%d}</small>' % (rng.randint(0, 9), rng.randint(0, 20)))
    L.append("      </footer>")
    L.append("    </div>")
    L.append("  );")
    L.append("}")
    return "\n".join(L) + "\n"


def _tsx_list(rng):
    """Sub-style 2: list-rendering component driven by rows.map()."""
    row_t = _pascal(rng, _UI_WORDS) + "Row"
    comp = _pascal(rng, _UI_WORDS) + "List"
    base = _kebab(rng, _UI_WORDS)
    rows, row = rng.choice([("rows", "row"), ("items", "item"),
                            ("entries", "entry"), ("records", "record")])
    numf = rng.choice(["count", "total", "score", "qty"])
    strf = rng.choice(["label", "name", "title", "slug"])

    L = []
    L.append('import React from "react";')
    L.append("")
    L.append("interface %s {" % row_t)
    L.append("  id: number;")
    L.append("  %s: string;" % strf)
    L.append("  %s: number;" % numf)
    L.append("}")
    L.append("")
    L.append("interface %sProps {" % comp)
    L.append("  %s: %s[];" % (rows, row_t))
    L.append("  onSelect?: (id: number) => void;")
    L.append("}")
    L.append("")
    L.append("export function %s({ %s, onSelect }: %sProps) {" % (comp, rows, comp))
    L.append("  const [active, setActive] = React.useState<number>(%d);" % rng.randint(0, 9))
    L.append("  return (")
    L.append('    <ul className="%s" data-testid="%s">' % (base, _kebab(rng, _UI_WORDS)))
    L.append("      {%s.map((%s) => (" % (rows, row))
    L.append("        <li")
    L.append("          key={%s.id}" % row)
    L.append('          className={%s.id === active ? "%s__item is-active" : "%s__item"}' % (row, base, base))
    L.append("          onClick={() => setActive(%s.id)}" % row)
    L.append("        >")
    L.append('          <span className="%s">{%s.%s}</span>' % (_kebab(rng, _UI_WORDS), row, strf))
    L.append('          <em className="%s">{%s.%s}</em>' % (rng.choice(_UI_WORDS), row, numf))
    if rng.random() < 0.75:
        L.append('          <button type="button" disabled={%s.%s === %d} onClick={() => onSelect?.(%s.id)}>'
                 % (row, numf, rng.choice([0, 1]), row))
        L.append("            %s" % _cap(rng.choice(_UI_WORDS)))
        L.append("          </button>")
    if rng.random() < 0.5:
        L.append('          <img src={"/thumbs/" + %s.%s + ".jpg"} alt={%s.%s} loading="lazy" />'
                 % (row, strf, row, strf))
    L.append("        </li>")
    L.append("      ))}")
    if rng.random() < 0.6:
        L.append("      {%s.length === 0 && (" % rows)
        L.append('        <li className="%s__empty">%s.</li>' % (base, _cap(_phrase(rng, 3))))
        L.append("      )}")
    L.append("    </ul>")
    L.append("  );")
    L.append("}")
    return "\n".join(L) + "\n"


def _tsx_form(rng):
    """Sub-style 3: form component with labeled inputs and a submit handler."""
    comp = _pascal(rng, _UI_WORDS) + "Form"
    base = _kebab(rng, _UI_WORDS)
    field = rng.choice(["email", "name", "query", "token", "phone"])
    setter = "set" + _cap(field)
    plan = rng.choice(["plan", "tier", "mode", "region"])

    L = []
    L.append('import React, { useState } from "react";')
    L.append("")
    L.append("interface %sProps {" % comp)
    L.append("  onSubmit: (value: string) => void;")
    L.append("  placeholder?: string;")
    L.append("}")
    L.append("")
    L.append("export const %s = ({ onSubmit, placeholder }: %sProps) => {" % (comp, comp))
    L.append('  const [%s, %s] = useState("");' % (field, setter))
    L.append("  return (")
    L.append('    <form className="%s" onSubmit={(e) => { e.preventDefault(); onSubmit(%s); }}>' % (base, field))
    L.append("      <fieldset>")
    L.append("        <legend>%s</legend>" % _cap(_phrase(rng, 2)))
    L.append('        <label htmlFor="%s">%s</label>' % (field, _cap(field)))
    L.append("        <input")
    L.append('          id="%s"' % field)
    L.append('          type="%s"' % rng.choice(["text", "email", "search", "tel"]))
    L.append("          value={%s}" % field)
    L.append("          onChange={(e) => %s(e.target.value)}" % setter)
    L.append("          placeholder={placeholder}")
    L.append("          maxLength={%d}" % rng.choice([32, 64, 128, 256]))
    L.append("          required")
    L.append("        />")
    L.append('        <select name="%s" defaultValue="%s">' % (plan, rng.choice(_FILLER_WORDS)))
    for _ in range(rng.randint(2, 4)):
        opt = rng.choice(_FILLER_WORDS)
        L.append('          <option value="%s">%s</option>' % (opt, _cap(opt)))
    L.append("        </select>")
    if rng.random() < 0.6:
        L.append('        <textarea name="%s" rows={%d} cols={%d} placeholder="%s" />'
                 % (rng.choice(_UI_WORDS), rng.randint(2, 8), rng.choice([32, 40, 60]), _phrase(rng, 2)))
    if rng.random() < 0.5:
        chk = rng.choice(_UI_WORDS)
        L.append("        <label>")
        L.append('          <input type="checkbox" name="%s" defaultChecked={%s} />' % (chk, rng.choice(["true", "false"])))
        L.append("          %s" % _cap(_phrase(rng, 2)))
        L.append("        </label>")
    L.append('        <button type="submit" className="btn btn--%s" disabled={%s.length === 0}>'
             % (rng.choice(_FILLER_WORDS), field))
    L.append("          %s" % _cap(rng.choice(["send", "save", "submit", "apply"])))
    L.append("        </button>")
    L.append("      </fieldset>")
    L.append("    </form>")
    L.append("  );")
    L.append("};")
    return "\n".join(L) + "\n"


def synth_hard_tsx(rng):
    """Return one hard TSX sample (bytes): JSX-dominated React component."""
    gen = rng.choice([_tsx_page, _tsx_list, _tsx_form])
    return gen(rng).encode("ascii")


# --------------------------------------------------------------------------
# Verilog / SystemVerilog sub-styles
# --------------------------------------------------------------------------

def _sv_uvm_class(rng):
    """Sub-style 1: UVM-flavored testbench classes (extends / new / virtual)."""
    item = _snake(rng, _HW_WORDS) + "_item"
    drv = _snake(rng, _HW_WORDS) + "_" + rng.choice(["driver", "monitor", "agent"])
    guard = item.upper() + "_SV"
    aw = rng.choice([4, 8, 12, 16])
    dw = rng.choice([8, 16, 32, 64])
    ifname = rng.choice(_HW_WORDS) + "_if"

    L = []
    L.append("`ifndef %s" % guard)
    L.append("`define %s" % guard)
    L.append("")
    L.append("class %s extends uvm_sequence_item;" % item)
    L.append("  rand bit [%d:0] addr;" % (aw - 1))
    L.append("  rand bit [%d:0] data;" % (dw - 1))
    L.append("  rand bit        wr_en;")
    L.append("")
    L.append("  constraint addr_c { addr inside {[0:%d]}; }" % (2 ** aw - rng.randint(1, 4)))
    L.append("")
    L.append("  function new(string name = \"%s\");" % item)
    L.append("    super.new(name);")
    L.append("  endfunction")
    L.append("")
    L.append("  virtual function string convert2string();")
    L.append('    return $sformatf("addr=%0h data=%0h wr=%0b", addr, data, wr_en);')
    L.append("  endfunction")
    L.append("endclass")
    L.append("")
    L.append("class %s extends uvm_driver #(%s);" % (drv, item))
    L.append("  `uvm_component_utils(%s)" % drv)
    L.append("")
    L.append("  virtual %s vif;" % ifname)
    L.append("  int unsigned %s_count;" % rng.choice(_HW_WORDS))
    L.append("")
    L.append("  function new(string name, uvm_component parent);")
    L.append("    super.new(name, parent);")
    L.append("  endfunction")
    L.append("")
    L.append("  virtual function void build_phase(uvm_phase phase);")
    L.append("    super.build_phase(phase);")
    L.append('    if (!uvm_config_db#(virtual %s)::get(this, "", "vif", vif))' % ifname)
    L.append('      `uvm_fatal("%s", "no virtual interface bound")' % drv.upper())
    L.append("  endfunction")
    L.append("")
    L.append("  virtual task run_phase(uvm_phase phase);")
    L.append("    forever begin")
    L.append("      seq_item_port.get_next_item(req);")
    L.append("      @(posedge vif.clk);")
    L.append("      vif.addr  <= req.addr;")
    L.append("      vif.data  <= req.data;")
    L.append("      vif.valid <= 1'b1;")
    if rng.random() < 0.7:
        L.append("      repeat (%d) @(posedge vif.clk);" % rng.randint(1, 4))
        L.append("      vif.valid <= 1'b0;")
    L.append("      seq_item_port.item_done();")
    L.append("    end")
    L.append("  endtask")
    L.append("endclass")
    L.append("")
    L.append("`endif // %s" % guard)
    return "\n".join(L) + "\n"


def _sv_typedef_pkg(rng):
    """Sub-style 2: package + interface header heavy on typedef struct packed."""
    pkg = _snake(rng, _HW_WORDS) + "_pkg"
    ifname = _snake(rng, _HW_WORDS) + "_if"
    dw = rng.choice([8, 16, 32, 64, 128])
    states = rng.sample(["IDLE", "SETUP", "BUSY", "WAIT", "DRAIN", "DONE", "ERROR"], rng.randint(3, 5))
    sw = max(1, (len(states) - 1).bit_length())

    L = []
    L.append("package %s;" % pkg)
    L.append("")
    L.append("  parameter int unsigned DATA_W = %d;" % dw)
    L.append("  parameter int unsigned %s_DEPTH = %d;" % (rng.choice(_HW_WORDS).upper(), rng.choice([4, 8, 16, 32])))
    L.append("")
    enum_body = ", ".join("%s = %d'd%d" % (s, sw, i) for i, s in enumerate(states))
    L.append("  typedef enum logic [%d:0] { %s } state_e;" % (sw - 1, enum_body))
    L.append("")
    for _ in range(rng.randint(1, 3)):
        st = _snake(rng, _HW_WORDS) + "_t"
        L.append("  typedef struct packed {")
        for _ in range(rng.randint(2, 5)):
            w = rng.choice([1, 2, 4, 8, 16, 32])
            f = _snake(rng, _HW_WORDS)
            if w == 1:
                L.append("    logic        %s;" % f)
            else:
                L.append("    logic [%2d:0] %s;" % (w - 1, f))
        L.append("  } %s;" % st)
        L.append("")
    L.append("  typedef logic [DATA_W-1:0] word_t;")
    L.append("")
    L.append("endpackage : %s" % pkg)
    L.append("")
    L.append("interface %s #(parameter int W = %d) (input logic clk, input logic rst_n);" % (ifname, dw))
    L.append("  logic [W-1:0] wdata;")
    L.append("  logic [W-1:0] rdata;")
    L.append("  logic         push;")
    L.append("  logic         pop;")
    L.append("  logic         full;")
    L.append("  logic         empty;")
    L.append("")
    L.append("  modport master (output wdata, push, pop, input rdata, full, empty);")
    L.append("  modport slave  (input  wdata, push, pop, output rdata, full, empty);")
    if rng.random() < 0.6:
        L.append("")
        L.append("  clocking cb @(posedge clk);")
        L.append("    default input #1step output #%dns;" % rng.randint(1, 3))
        L.append("    output wdata, push;")
        L.append("    input  full, empty;")
        L.append("  endclocking")
    L.append("endinterface : %s" % ifname)
    return "\n".join(L) + "\n"


def _sv_module(rng):
    """Sub-style 3: synthesizable parameterized module with always_ff."""
    mod = _snake(rng, _HW_WORDS)
    dw = rng.choice([8, 16, 32, 64])
    depth = rng.choice([4, 8, 16, 32, 64])
    cnt = rng.choice(_HW_WORDS) + "_cnt"

    L = []
    L.append("module %s #(" % mod)
    L.append("  parameter int DATA_W = %d," % dw)
    L.append("  parameter int DEPTH  = %d" % depth)
    L.append(") (")
    L.append("  input  logic              clk,")
    L.append("  input  logic              rst_n,")
    L.append("  input  logic [DATA_W-1:0] din,")
    L.append("  input  logic              push,")
    L.append("  input  logic              pop,")
    L.append("  output logic [DATA_W-1:0] dout,")
    L.append("  output logic              full,")
    L.append("  output logic              empty")
    L.append(");")
    L.append("")
    L.append("  logic [DATA_W-1:0] mem [DEPTH];")
    L.append("  logic [$clog2(DEPTH)-1:0] wptr, rptr;")
    L.append("  logic [$clog2(DEPTH):0]   %s;" % cnt)
    L.append("")
    L.append("  always_ff @(posedge clk or negedge rst_n) begin")
    L.append("    if (!rst_n) begin")
    L.append("      wptr <= '0;")
    L.append("      rptr <= '0;")
    L.append("      %s <= '0;" % cnt)
    L.append("    end else begin")
    L.append("      if (push && !full) begin")
    L.append("        mem[wptr] <= din;")
    L.append("        wptr <= wptr + 1'b1;")
    L.append("      end")
    L.append("      if (pop && !empty) begin")
    L.append("        rptr <= rptr + 1'b1;")
    L.append("      end")
    L.append("      %s <= %s + (push && !full) - (pop && !empty);" % (cnt, cnt))
    L.append("    end")
    L.append("  end")
    L.append("")
    L.append("  assign dout  = mem[rptr];")
    L.append("  assign full  = (%s == DEPTH);" % cnt)
    L.append("  assign empty = (%s == '0);" % cnt)
    if rng.random() < 0.7:
        st = rng.choice(_HW_WORDS) + "_q"
        L.append("")
        L.append("  logic [%d:0] %s;" % (rng.choice([3, 7, 15]), st))
        L.append("  always_ff @(posedge clk) begin")
        L.append("    if (push) %s <= %s ^ {%s[%d:0], din[0]};" % (st, st, st, rng.choice([2, 6, 14])))
        L.append("  end")
    if rng.random() < 0.5:
        L.append("")
        L.append("`ifndef SYNTHESIS")
        L.append("  initial begin")
        L.append('    $display("%s: DATA_W=%%0d DEPTH=%%0d", DATA_W, DEPTH);' % mod)
        L.append("  end")
        L.append("`endif")
    L.append("")
    L.append("endmodule : %s" % mod)
    return "\n".join(L) + "\n"


def synth_hard_verilog(rng):
    """Return one hard SystemVerilog sample (bytes): OOP/typedef/module mix."""
    gen = rng.choice([_sv_uvm_class, _sv_typedef_pkg, _sv_module])
    return gen(rng).encode("ascii")


# --------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    rng = random.Random(0)
    sep = "=" * 40
    for fn in (synth_hard_tsx, synth_hard_verilog):
        for _ in range(2):
            print(sep)
            print(fn(rng).decode("ascii"))
    print(sep)
