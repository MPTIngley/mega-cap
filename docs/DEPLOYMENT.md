# StockPulse Deployment Guide

This guide covers process supervision for the StockPulse scheduler and dashboard.

## Option 1: systemd (Recommended for Linux servers)

### Install the service files

```bash
sudo cp deploy/stockpulse.service /etc/systemd/system/
sudo cp deploy/stockpulse-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### Enable and start services

```bash
sudo systemctl enable stockpulse
sudo systemctl enable stockpulse-dashboard
sudo systemctl start stockpulse
sudo systemctl start stockpulse-dashboard
```

### Manage services

```bash
sudo systemctl status stockpulse
sudo systemctl status stockpulse-dashboard

sudo systemctl restart stockpulse
sudo systemctl stop stockpulse

sudo journalctl -u stockpulse -f
```

### Service features
- Auto-restart on crash (after 10 seconds)
- Starts on boot
- Rate limiting: max 5 restarts in 5 minutes
- Logs to `/home/user/mega-cap/logs/`

---

## Option 2: Supervisor

### Install supervisor

```bash
pip install supervisor
```

### Start with supervisor

```bash
supervisord -c /home/user/mega-cap/deploy/supervisord.conf
```

### Manage processes

```bash
supervisorctl -c /home/user/mega-cap/deploy/supervisord.conf status
supervisorctl -c /home/user/mega-cap/deploy/supervisord.conf restart stockpulse
supervisorctl -c /home/user/mega-cap/deploy/supervisord.conf restart dashboard
supervisorctl -c /home/user/mega-cap/deploy/supervisord.conf stop all
supervisorctl -c /home/user/mega-cap/deploy/supervisord.conf start all
```

### Supervisor features
- Auto-restart on crash
- Log rotation (50MB, 5 backups)
- Both scheduler and dashboard managed together

---

## Quick Reference

| Action | systemd | supervisor |
|--------|---------|------------|
| Start | `systemctl start stockpulse` | `supervisorctl start stockpulse` |
| Stop | `systemctl stop stockpulse` | `supervisorctl stop stockpulse` |
| Restart | `systemctl restart stockpulse` | `supervisorctl restart stockpulse` |
| Status | `systemctl status stockpulse` | `supervisorctl status` |
| Logs | `journalctl -u stockpulse -f` | `tail -f logs/stockpulse.log` |

---

## Log Files

All logs are written to `/home/user/mega-cap/logs/`:
- `stockpulse.log` - Main scheduler logs
- `dashboard.log` - Streamlit dashboard logs
- `supervisord.log` - Supervisor daemon logs (if using supervisor)
