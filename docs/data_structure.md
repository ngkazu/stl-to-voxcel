# Voxel データ構造 設計書

本ドキュメントは stl-to-voxcel のコア Voxel データ構造を説明する。

## 概要

STL をボクセル化した結果は 8 分木（octree）で表現する。
各セル（`Cell`）は **locational code（位置コード）** と呼ばれる一意な整数 ID を持つ。
locational code は「ルートからそのセルへ至る 8 分木の経路」をビット列で符号化したもので、
親子関係・深さ・座標をすべてこの ID から計算できる。

## 用語

| 用語 | 意味 |
|------|------|
| セル (Cell) | 8 分木の 1 ノード（確定した 1 ボクセル） |
| ルート | 8 分木の最上位ノード（STL 全体の bbox） |
| depth | ルートからの分割回数（ルート = 0） |
| 子番号 (octant) | 親を 8 分割したときの各子の番号 0〜7 |
| locational code | ルートからの経路を符号化した整数 ID |

## locational code（方式2: 番兵ビット付き）

### 定義

ルートを `1` とし、子へ下るたびに「3 ビットの子番号」を右に連結する。

```
depth 0 (ルート):  1
depth 1:           1 <c1>            (3bit)
depth 2:           1 <c1> <c2>       (6bit)
depth D:           1 <c1> <c2> ... <cD>   (3D bit)
```

- 先頭の `1` は **番兵ビット**。これにより先行ゼロの経路（子番号 0 が続く経路）も
  一意に識別でき、ID 単体から depth が判定できる。
- 各 `<ci>` は 0〜7 の子番号（3 ビット）。

### 子番号のビット割り当て

子番号は 8 分割時の各軸の上下（0 = 下半分 / 1 = 上半分）から決まる。

```
octant = (ix << 2) | (iy << 1) | iz
```

| 子番号 | ix (X) | iy (Y) | iz (Z) | ビット |
|:------:|:------:|:------:|:------:|:------:|
| 0 | 0 | 0 | 0 | 000 |
| 1 | 0 | 0 | 1 | 001 |
| 2 | 0 | 1 | 0 | 010 |
| 3 | 0 | 1 | 1 | 011 |
| 4 | 1 | 0 | 0 | 100 |
| 5 | 1 | 0 | 1 | 101 |
| 6 | 1 | 1 | 0 | 110 |
| 7 | 1 | 1 | 1 | 111 |

- `ix=1` は X 軸で上半分（origin_x + size_x/2 側）
- `iy=1` は Y 軸で上半分
- `iz=1` は Z 軸で上半分

この割り当ては `octree._split_octants` の分割順序（`for ix: for iy: for iz:`）と一致する。

### 例（あなたの「遺伝子ID」イメージとの対応）

内部表現はビット（8 進表示が読みやすい）：

```
ルート          : 1          (2進: 1,        8進: 1)
depth1 子0       : 1 000      (2進: 1000,     8進: 10)
depth1 子7       : 1 111      (2進: 1111,     8進: 17)
depth2 子1→子5   : 1 001 101  (2進: 1001101,  8進: 115)
```

8 進で表示すると `1, 10, 17, 115 ...` のように「ルート 1 → 各階層の子番号」が
そのまま桁として読める。

## locational code からの導出

ルートの `origin`（= bbox_min）と `base_cell_size`（= ルート 1 辺）は
モデル全体で 1 つだけ保持する（セルごとには持たない）。

### depth の取得

```
depth = (bit_length(code) - 1) / 3
```

例: `code = 0b1001101`（7bit） → depth = (7 - 1) / 3 = 2

### 親の取得

```
parent(code) = code >> 3      （ルート code=1 で停止）
```

### 子の取得

```
child(code, octant) = (code << 3) | octant      （octant: 0〜7）
```

### origin / size の復元

ルートから子番号を順にたどり、各軸を半分ずつ絞り込む。

```
size  = base_cell_size / (2 ** depth)       （軸ごと: base_size_axis / 2**depth）
origin = bbox_min + Σ (octant_bit_axis * size_at_level)
```

擬似コード：

```python
def decode(code, bbox_min, base_cell_size):
    depth = (code.bit_length() - 1) // 3
    # 番兵ビットを除いた経路部分を取り出す
    path = code & ((1 << (3 * depth)) - 1)
    origin = bbox_min.astype(float).copy()
    size = base_cell_size.astype(float).copy()   # [sx, sy, sz]
    for level in range(depth):
        size = size / 2.0
        # 上位階層から順に 3bit ずつ読む
        shift = 3 * (depth - 1 - level)
        octant = (path >> shift) & 0b111
        ix = (octant >> 2) & 1
        iy = (octant >> 1) & 1
        iz = octant & 1
        origin += np.array([ix, iy, iz]) * size
    return origin, size
```

> 注: `base_cell_size` は cubic_root の有無で立方体 / 非立方体が変わるため `[sx, sy, sz]` の
> ベクトルで持つ。各軸は分割のたびに半分になる。

## Cell データ構造

```python
class CellState(IntEnum):
    EMPTY = 0    # 空
    SHELL = 1    # 表面（STL と交差）
    INSIDE = 2   # 内部（充填）
    OUTSIDE = 3  # 外部（expand 追加分）

@dataclass
class Cell:
    code: int                       # locational code（方式2）
    state: CellState                # セルの状態
    facet_indices: tuple[int, ...] | None = None   # 交差する STL 面番号（SHELL のみ）
    # origin / size / depth は code から計算で復元（フィールドとして持たない）
```

- `depth` は `code` から計算できるため**フィールドとして持たない**。
- `origin` / `size` も `code` + モデルの `bbox_min` / `base_cell_size` から復元できる。
- メモリは code（int 1 個）+ state + 任意の facet_indices のみ。

## VoxelModel データ構造

```python
@dataclass
class VoxelModel:
    bbox_min: np.ndarray        # ルートの origin（= STL bbox 最小）
    base_cell_size: np.ndarray  # ルート 1 辺 [sx, sy, sz]
    cells: dict[int, Cell]      # locational code -> Cell（単一 dict で全 depth を管理）
```

- セルは `dict[code -> Cell]` で保持。方式2 は code 単体で一意なので単一 dict で済む。
- 座標復元に必要な `bbox_min` / `base_cell_size` はモデルが 1 つだけ持つ。

## 検索操作（locational code の利点）

| 操作 | 方法 | 計算量 |
|------|------|--------|
| 親検索 | `code >> 3` | O(1) |
| 子検索 | `(code << 3) \| octant` | O(1) |
| depth 判定 | `(bit_length - 1) / 3` | O(1) |
| 存在確認 | `code in cells` | O(1) 平均 |
| 隣接検索 | locational code 演算（Samet 法） | O(depth) |

### 隣接検索（概要）

あるセルの指定軸・方向の隣接セルを求める：

1. 現セルの code から、指定方向に隣接する「同一 depth の候補 code」を計算する
   （Samet のアルゴリズム: 境界をまたぐ場合は上位階層へ桁上がり）。
2. 候補 code が `cells` に存在すれば、それが同サイズの隣接セル。
3. 存在しなければ `code >> 3` で親を辿る → より大きい隣接セル。
4. より小さい隣接セル（子側）が必要なら、候補 code の子を展開する。

> 隣接探索の詳細アルゴリズムは shrink / expand 実装時に別途詰める。

## 設計の狙い

- **8 分木を維持**（疎な領域を大セルで表現でき、メモリ効率が良い）。
- **セルの履歴（親子）が ID から辿れる**（遺伝子的 ID）。
- **depth / origin / size を ID から計算**するため、セルごとのメモリは最小。
- **隣接検索・存在確認が効率的**（shrink / expand の基盤）。

## 今後の予定

- shrink（縮小）機能: 外殻面を STL 表面まで平行移動させる。
  面ごとの移動後位置は別途 `face_positions` 等で保持する（本書では未定義）。
- expand（膨張）機能: 外側領域を ID 体系へ収める方法を別途検討。
