#!/bin/bash

# Deploy AlphaScrabble to Production

set -e

echo "🚀 Deploying AlphaScrabble to Production..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "docker-compose.prod.yml" ]; then
    echo "❌ Error: docker-compose.prod.yml not found. Please run from project root."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models data ssl

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🏥 Checking service health..."
if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
    echo "✅ Services are running successfully!"
else
    echo "❌ Some services failed to start. Check logs with:"
    echo "   docker-compose -f docker-compose.prod.yml logs"
    exit 1
fi

# Display service information
echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Service Information:"
echo "   Web Application: http://localhost"
echo "   API Endpoint: http://localhost/api"
echo "   Health Check: http://localhost/health"
echo ""
echo "🔧 Management Commands:"
echo "   View logs: docker-compose -f docker-compose.prod.yml logs"
echo "   Stop services: docker-compose -f docker-compose.prod.yml down"
echo "   Restart services: docker-compose -f docker-compose.prod.yml restart"
echo "   Update services: docker-compose -f docker-compose.prod.yml pull && docker-compose -f docker-compose.prod.yml up -d"
echo ""
echo "📈 Monitoring:"
echo "   Service status: docker-compose -f docker-compose.prod.yml ps"
echo "   Resource usage: docker stats"
echo ""
echo "🔒 Security Notes:"
echo "   - Change default passwords in production"
echo "   - Set up SSL certificates for HTTPS"
echo "   - Configure firewall rules"
echo "   - Enable log monitoring"
echo ""
echo "Ready to play AlphaScrabble! 🎮"
