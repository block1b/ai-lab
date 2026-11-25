// stage2 CLI 入口。
//
//   go run ./stage2/cmd "屏幕很好但电池太差"   # 单次抽取（需 API）
//   go run ./stage2/cmd --demo                 # 本地 Schema 校验演示（零成本）
//   go run ./stage2/cmd --eval 20              # 批量验收（需 API）
package main

import "golab/stage2"

func main() { stage2.Main() }
