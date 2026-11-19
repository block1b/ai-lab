// Package device 提供内存版的设备与告警数据源，供 MCP 工具调用。
// 真实项目里这里应是 MySQL/MongoDB 的访问层，示例中用内存实现保证可直接运行。
package device

import (
	"errors"
	"sort"
	"sync"
	"time"
)

// Status 设备状态
type Status string

const (
	StatusOnline    Status = "online"
	StatusOffline   Status = "offline"
	StatusRebooting Status = "rebooting"
)

// ErrNotFound 设备不存在（或不属于该租户）
var ErrNotFound = errors.New("device not found")

// Device 设备
type Device struct {
	ID       string    `json:"id"`
	Name     string    `json:"name"`
	Tenant   string    `json:"tenant"`
	Model    string    `json:"model"`
	IP       string    `json:"ip"`
	Status   Status    `json:"status"`
	LastSeen time.Time `json:"last_seen"`
}

// Alarm 设备告警
type Alarm struct {
	ID        string    `json:"id"`
	DeviceID  string    `json:"device_id"`
	Tenant    string    `json:"tenant"`
	Level     string    `json:"level"` // info / warn / critical
	Message   string    `json:"message"`
	CreatedAt time.Time `json:"created_at"`
}

// Store 内存数据源，读写加锁保证并发安全
type Store struct {
	mu      sync.RWMutex
	devices map[string]*Device
	alarms  []Alarm
}

// NewStore 构造带示例数据的 Store
func NewStore() *Store {
	now := time.Now()
	s := &Store{devices: map[string]*Device{}}

	seed := []Device{
		{ID: "dev-1001", Name: "三号楼门禁-01", Tenant: "zzu", Model: "POS-A6", IP: "10.12.1.11", Status: StatusOnline},
		{ID: "dev-1002", Name: "三号楼门禁-02", Tenant: "zzu", Model: "POS-A6", IP: "10.12.1.12", Status: StatusOffline},
		{ID: "dev-1003", Name: "图书馆闸机-01", Tenant: "zzu", Model: "GATE-X2", IP: "10.12.2.21", Status: StatusOnline},
		{ID: "dev-1004", Name: "南门人脸一体机", Tenant: "zzu", Model: "FACE-S3", IP: "10.12.3.31", Status: StatusOffline},
		{ID: "dev-1005", Name: "宿舍水电表-07", Tenant: "zzu", Model: "METER-W1", IP: "10.12.4.41", Status: StatusOnline},
		{ID: "dev-2001", Name: "实验楼门禁-01", Tenant: "haut", Model: "POS-A6", IP: "10.20.1.11", Status: StatusOnline},
		{ID: "dev-2002", Name: "食堂消费机-03", Tenant: "haut", Model: "POS-P8", IP: "10.20.2.13", Status: StatusOffline},
	}
	for i := range seed {
		seed[i].LastSeen = now.Add(-time.Duration(i+1) * 7 * time.Minute)
		s.devices[seed[i].ID] = &seed[i]
	}

	s.alarms = []Alarm{
		{ID: "al-9001", DeviceID: "dev-1002", Tenant: "zzu", Level: "critical", Message: "设备离线超过 30 分钟，心跳连续丢失", CreatedAt: now.Add(-18 * time.Minute)},
		{ID: "al-9002", DeviceID: "dev-1004", Tenant: "zzu", Level: "critical", Message: "人脸比对服务无响应", CreatedAt: now.Add(-9 * time.Minute)},
		{ID: "al-9003", DeviceID: "dev-1001", Tenant: "zzu", Level: "warn", Message: "CPU 使用率持续高于 85%", CreatedAt: now.Add(-42 * time.Minute)},
		{ID: "al-9004", DeviceID: "dev-2002", Tenant: "haut", Level: "critical", Message: "设备离线，最后一次心跳 2 小时前", CreatedAt: now.Add(-96 * time.Minute)},
		{ID: "al-9005", DeviceID: "dev-2001", Tenant: "haut", Level: "info", Message: "固件升级完成", CreatedAt: now.Add(-25 * time.Minute)},
	}
	return s
}

// ListDevices 按租户与状态查询设备；status 为空表示不限
func (s *Store) ListDevices(tenant, status string) []Device {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]Device, 0, len(s.devices))
	for _, d := range s.devices {
		if tenant != "" && d.Tenant != tenant {
			continue
		}
		if status != "" && string(d.Status) != status {
			continue
		}
		out = append(out, *d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// QueryAlarms 查询 since 之后、指定级别的告警；level 为空表示不限
func (s *Store) QueryAlarms(tenant string, since time.Time, level string) []Alarm {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]Alarm, 0, len(s.alarms))
	for _, a := range s.alarms {
		if tenant != "" && a.Tenant != tenant {
			continue
		}
		if level != "" && a.Level != level {
			continue
		}
		if a.CreatedAt.Before(since) {
			continue
		}
		out = append(out, a)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt.After(out[j].CreatedAt) })
	return out
}

// Restart 重启设备：校验租户归属，避免越权操作
func (s *Store) Restart(deviceID, tenant string) (Device, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	d, ok := s.devices[deviceID]
	if !ok || (tenant != "" && d.Tenant != tenant) {
		return Device{}, ErrNotFound
	}
	d.Status = StatusRebooting
	d.LastSeen = time.Now()
	return *d, nil
}
