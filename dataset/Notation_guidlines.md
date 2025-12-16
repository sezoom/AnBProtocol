# Create the Markdown file with the unified style guide


---

## 1) File Skeleton & Section Order

```text
Protocol <Name>:

Declarations:
    <symbol>/<arity>;
    ...

Types:
    Agent A,B[, ...];
    Number <list of atoms>;
    Symmetric_key <list>;
    Function <list of function symbols>;
    Mapping <list of mapping symbols>

Knowledge:
    A : <comma-separated atoms>;
    B : <...>;
    [other roles...]

Public:
    <comma-separated public atoms>;

Private:
    <comma-separated private atoms>;

Actions:
    [label] Sender -> Receiver (FreshVars) : <message term>;
    ...

Goals:
    <goal lines>;

ChannelKeys:
    K(X,Y): <term>;
    ...

end
```

- **Required:** `Protocol`, `Declarations`, `Types`, `Knowledge`, `Actions`, `end`  
- **Optional:** `Public`, `Private`, `Goals`, `ChannelKeys`

---

## 2) Declarations (Function Symbols)

- List **all** term constructors with **arity**:
  - `aenc/2`, `senc/2`, `hash/1` or `h/1`, `prf/3`, `pre/1`, `g/0`, etc.
- Declare only **symbols** here (no agents/variables).

**Example**
```text
Declarations:
    aenc/2;
    senc/2;
    h/1;
    g/0;
```

---

## 3) Types

- **Agent:** role/principal atoms, e.g., `Agent A,B,S;`
- **Number:** nonces, tags, SIDs, payload atoms, exponents, e.g., `Number NA,NB,Sid,tagX1;`
- **Symmetric_key:** named shared keys: `Symmetric_key KAB,KCG,KCS;`
- **Function:** uninterpreted functions: `Function h,prf,pre,xor;`
- **Mapping:** maps like `pk/1`, `sk/1` (or `sk/2` for shared secrets)

Any function other than aenc,senc shall be declared in Functions.
**Example**
```text
Types:
    Agent A,B,S;
    Number NA,NB,Sid,tagP1,tagP2;
    Data M1,M2;
    Symmetric_key KAB;
    Function h,prf,pre, xor;
    Mapping pk,sk
```

---

## 4) Knowledge

- For each role, list initial knowledge as **atoms/terms** (no message structures).

**Example**
```text
Knowledge:
    A : A, B, S, pk(A), sk(A), pk(B), B, pk(S), NA0;
    B : A, B,S, pk(B), sk(B), pk(A), pk(S);
    S : S, sk(S), pk(S), A, B, pk(A), pk(B);
```

---

## 5) Public / Private

- **Public:** terms initially known to the adversary (e.g., public keys, identities).
- **Private:** terms initially unknown to the adversary (e.g., private keys, secret keys).
- Keep sections empty if unused, but preserve headers.

---

## 6) Actions (Messages)

**Canonical line**
```text
[label] Sender -> Receiver (Fresh1,Fresh2,...) : <term>
```

- **Label:** `[m1]`, `[step2]`, etc.
- **Fresh list (optional):** variables freshly generated **by the sender** in this step.
- **Message term:**
  - **Concatenation:** `.` (left-associative), e.g., `A . B . NA`
  - **Asymmetric:** `aenc{ <term> }pk(B)` or signature-style `aenc{ <term> }sk(A)`
  - **Symmetric:** `senc{ <term> }K` or `senc{ <term> }sk(A,B)`
  - **Hash:** `h(<term>)` (or `hash(<term>)`)
  - **Nesting:** allowed (e.g., `aenc{ X . senc{Y}K }pk(B)`)
  - **DH/exponent:** `g()^na`, `(g()^(na)^nb)` (parenthesize as needed)
- **Multiple payloads:** prefer a single concatenated term using `.`

**Examples**
```text
[m1] A -> B (NA)        : aenc{ A . pk(A) . NA }sk(A);
[m2] B -> A (NB)        : aenc{ B . pk(B) . NB }pk(A);
[m3] A -> B             : senc{ h(NA . NB) }KAB;
```

---

## 7) Tags & Sessions

- **Tags:** atoms under `Number` (e.g., `tagTLS3`, `tagANSCK4`) included as concatenands: `tagTLS3 . ...`
- **Session IDs:** declare `Number Sid` and include in all messages for binding (optional):
  `A . B . Sid . NA . ...`

---

## 8) Goals

**Supported forms**
- **Authentication / Agreement**
    - `B -> A : X;` (B authenticates A on X injective by default)
    - `B ->* A : X;` (B authenticates A on X non-injective)
    - `B *->* A : X;` (B authenticates A on X mutual non-injective pattern)
- **Secrecy**
  - `X secret between A,B;`

**Example**
```text
Goals:
   /* B authenticates A on prf(PMS,NA,NB) */
   B -> A : prf(PMS,NA,NB);
   prf(PMS,NA,NB) secret between A,B;
```

---

## 9) ChannelKeys

- Declare **directional** derived keys if modeled:
```text
ChannelKeys:
    K(A,B): clientK(NA,NB,prf(PMS,NA,NB));
    K(B,A): serverK(NA,NB,prf(PMS,NA,NB));
```
- Omit if not using directional channels.

---

## 10) Identifiers & Naming

- **Agents:** `A,B,S` or descriptive role names
- **Nonces/Numbers:** `NA, NB, n, n_1, nb0`
- **Tags:** `tag<Proto><#>` (e.g., `tagTLS1`)
- **Keys:** `sk(A,B)`, `KAB`; public/private as `pk(A)`, `sk(A)`
- **Functions:** `f`, `pre`, `h`, `mac`

---

## 11) Comments

- Use block `/* ... */` or hash `# ...` comments **outside** terms/messages.

**Examples**
```text
# Verified
/* Goals that do not hold */
```

---

## 12) Formatting Rules

- One declaration per line, each ending with `;`
- Spaces:
  - around `->`
  - after commas in fresh lists
- `.` used solely as concatenation separator
- Consistent indentation (4 spaces inside sections)
- Optional blank line between sections
- Always end the file with a standalone `end`

---

## 13) Minimal Template

```text
Protocol ExampleProto:

Declarations:
    aenc/2;
    senc/2;
    h/1;

Types:
    Agent A,B,S;
    Number NA,NB,Sid,tagE1,tagE2,tagE3;
    Symmetric_key KAB;
    Function h,prf;
    Mapping pk,sk

Knowledge:
    A : pk(A), sk(A), pk(B), A,B;
    B : pk(B), sk(B), pk(A), A,B;

Public:
    A,B,pk(A), pk(B);

Private:
    sk(A), sk(B), KAB;

Actions:
    [e1] A -> B (NA) : aenc{ A . B . Sid . NA . tagE1 }sk(A);
    [e2] B -> A (NB) : aenc{ B . A . Sid . NB . tagE2 }pk(A);
    [e3] A -> B      : senc{ h(NA . NB . Sid . tagE3) }KAB;

Goals:
    B -> A : h(NA . NB . Sid);
    h(NA . NB . Sid) secret between A,B;

ChannelKeys:
    K(A,B): prf(NA,NB,Sid);

end
```

---

## 14) Validation Checklist (Quick)

- [ ] All function symbols are in `Declarations` with correct arity  
- [ ] All atoms/variables declared under correct **Type**  
- [ ] Every key used (`pk`, `sk`, `K...`) is declared/typed  
- [ ] Fresh variables in `Actions` are declared in `Types:Number`  
- [ ] `Public`/`Private` list only atoms/terms (no message structures)  
- [ ] Each action matches `[label] X -> Y (Fresh?) : <term>;`  
- [ ] Goals use canonical lines and end with `;`  
- [ ] File ends with `end`
