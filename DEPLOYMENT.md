# AlphaScrabble Deployment Guide

This guide covers deploying AlphaScrabble to production environments.

## 🚀 Quick Deployment

### Local Development
```bash
# Start web server for development
./scripts/start_web.sh
```

### Production Deployment
```bash
# Deploy to production
./scripts/deploy.sh
```

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: 2 cores minimum, 4 cores recommended
- **Storage**: 10GB free space
- **Network**: Internet connection for package downloads

### Software Requirements
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Python**: 3.10+ (for development)
- **Git**: Latest version

### Installation Commands
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose python3 python3-pip git

# CentOS/RHEL
sudo yum install -y docker docker-compose python3 python3-pip git
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
```

## 🏗️ Architecture

### Production Stack
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │  AlphaScrabble  │    │     Redis       │
│   (Load Balancer)│    │   (Web App)     │    │   (Cache)       │
│     Port 80     │────│    Port 5000    │    │   Port 6379     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │    PostgreSQL   │
                       │   (Database)    │
                       │   Port 5432     │
                       └─────────────────┘
```

### Components
- **Nginx**: Reverse proxy and load balancer
- **AlphaScrabble Web**: Flask application with AI engine
- **Redis**: Caching and session storage
- **PostgreSQL**: Game data and user management

## 🔧 Configuration

### Environment Variables
```bash
# Production settings
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:pass@postgres:5432/alphascrabble
REDIS_URL=redis://redis:6379/0
```

### Docker Compose Configuration
The production deployment uses `docker-compose.prod.yml` with:
- **Resource limits**: Memory and CPU constraints
- **Health checks**: Automatic service monitoring
- **Restart policies**: Automatic recovery
- **Volume mounts**: Persistent data storage

## 📊 Monitoring

### Health Checks
```bash
# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs

# Check resource usage
docker stats
```

### Application Health
- **Health endpoint**: `http://localhost/health`
- **API status**: `http://localhost/api/status`
- **Game statistics**: `http://localhost/api/stats`

## 🔒 Security

### Production Security Checklist
- [ ] Change default passwords
- [ ] Set up SSL certificates
- [ ] Configure firewall rules
- [ ] Enable log monitoring
- [ ] Set up backup procedures
- [ ] Configure rate limiting
- [ ] Enable security headers

### SSL Configuration
```bash
# Generate SSL certificates (Let's Encrypt)
sudo apt install certbot
sudo certbot certonly --standalone -d yourdomain.com

# Copy certificates to ssl directory
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/key.pem
sudo chown $USER:$USER ssl/*.pem
```

### Firewall Configuration
```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# iptables (CentOS)
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -j DROP
```

## 📈 Scaling

### Horizontal Scaling
```yaml
# docker-compose.prod.yml
services:
  alphascrabble-web:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
```

### Load Balancing
```nginx
# nginx.conf
upstream alphascrabble {
    server alphascrabble-web-1:5000;
    server alphascrabble-web-2:5000;
    server alphascrabble-web-3:5000;
}
```

### Database Scaling
- **Read replicas**: For read-heavy workloads
- **Connection pooling**: PgBouncer for connection management
- **Caching**: Redis for frequently accessed data

## 🔄 Updates

### Rolling Updates
```bash
# Update application
git pull origin main
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Zero-downtime deployment
docker-compose -f docker-compose.prod.yml up -d --no-deps alphascrabble-web
```

### Backup and Recovery
```bash
# Backup database
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U alphascrabble alphascrabble > backup.sql

# Restore database
docker-compose -f docker-compose.prod.yml exec -T postgres psql -U alphascrabble alphascrabble < backup.sql
```

## 🐛 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs alphascrabble-web

# Check resource usage
docker stats

# Restart service
docker-compose -f docker-compose.prod.yml restart alphascrabble-web
```

#### Database Connection Issues
```bash
# Check database status
docker-compose -f docker-compose.prod.yml exec postgres pg_isready

# Test connection
docker-compose -f docker-compose.prod.yml exec postgres psql -U alphascrabble -d alphascrabble -c "SELECT 1;"
```

#### Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.prod.yml
# Restart services
docker-compose -f docker-compose.prod.yml up -d
```

### Performance Optimization

#### Application Performance
- **Enable gzip compression** in Nginx
- **Use Redis caching** for frequently accessed data
- **Optimize database queries** with proper indexing
- **Enable connection pooling** for database connections

#### System Performance
- **Use SSD storage** for better I/O performance
- **Enable swap** for memory management
- **Monitor resource usage** with tools like htop
- **Set up log rotation** to prevent disk space issues

## 📞 Support

### Getting Help
- **Documentation**: Check this guide and README.md
- **Issues**: Report bugs on GitHub Issues
- **Community**: Join discussions on GitHub Discussions

### Log Locations
- **Application logs**: `docker-compose -f docker-compose.prod.yml logs`
- **System logs**: `/var/log/syslog`
- **Nginx logs**: `docker-compose -f docker-compose.prod.yml logs nginx`

### Performance Monitoring
- **Application metrics**: Built-in health endpoints
- **System metrics**: Use tools like Prometheus + Grafana
- **Log analysis**: Use ELK stack (Elasticsearch, Logstash, Kibana)

## 🎯 Production Checklist

Before going live:
- [ ] All services are running and healthy
- [ ] SSL certificates are configured
- [ ] Firewall rules are set up
- [ ] Backup procedures are in place
- [ ] Monitoring is configured
- [ ] Log rotation is enabled
- [ ] Security headers are configured
- [ ] Rate limiting is enabled
- [ ] Error handling is tested
- [ ] Performance is optimized

Ready to deploy AlphaScrabble to production! 🚀
