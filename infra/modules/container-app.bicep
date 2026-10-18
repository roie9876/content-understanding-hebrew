param appName string
param environmentName string
param containerImage string
param location string
param tags object
param acrName string
param managedIdentityResourceId string
param managedIdentityClientId string
param foundryOpenAiEndpoint string
param contentUnderstandingEndpoint string
param storageName string
param demoExpiresAt string

var placeholderImage = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
var isPlaceholder = containerImage == placeholderImage
var effectivePort = isPlaceholder ? 80 : 8080
var effectiveProbePath = isPlaceholder ? '/' : '/healthz'

resource environment 'Microsoft.App/managedEnvironments@2026-07-01' existing = {
  name: environmentName
}

resource registry 'Microsoft.ContainerRegistry/registries@2025-11-01' existing = {
  name: acrName
}

resource app 'Microsoft.App/containerApps@2026-07-01' = {
  name: appName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentityResourceId}': {}
    }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: effectivePort
        allowInsecure: false
        transport: 'auto'
      }
      registries: isPlaceholder ? [] : [
        {
          server: registry.properties.loginServer
          identity: managedIdentityResourceId
        }
      ]
      secrets: []
    }
    template: {
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
      containers: [
        {
          name: 'inksight'
          image: containerImage
          resources: {
            cpu: 2
            memory: '4Gi'
          }
          env: concat(
            [
              { name: 'PORT', value: string(effectivePort) }
              { name: 'PUBLIC_DEMO', value: '1' }
              { name: 'ENABLE_GOLDEN_COMPARISON', value: '0' }
              { name: 'INKSIGHT_DATA_DIR', value: '/data' }
              { name: 'INKSIGHT_RESULTS_DIR', value: '/data/analysis_results' }
              { name: 'DEMO_MAX_UPLOAD_BYTES', value: '20971520' }
              { name: 'DEMO_MAX_PAGES', value: '2' }
              { name: 'DEMO_MAX_REGIONS', value: '40' }
              { name: 'DEMO_MAX_JOBS_PER_HOUR', value: '10' }
              { name: 'DEMO_MAX_JOBS_TOTAL', value: '25' }
              { name: 'DEMO_EXPIRES_AT', value: demoExpiresAt }
              { name: 'AZURE_AUTH_MODE', value: 'managed-identity' }
              { name: 'AZURE_CLIENT_ID', value: managedIdentityClientId }
              { name: 'AZURE_TENANT_ID', value: subscription().tenantId }
              { name: 'MODEL_BACKEND', value: 'azure-responses' }
              { name: 'GPT_ENDPOINT', value: foundryOpenAiEndpoint }
              { name: 'GPT_DEPLOYMENT', value: 'gpt-6.1-sol-c280' }
              { name: 'GPT_61_SOL_ENDPOINT', value: foundryOpenAiEndpoint }
              { name: 'GPT_61_SOL_DEPLOYMENT', value: 'gpt-6.1-sol-c280' }
              { name: 'GPT_TARGETS', value: '[]' }
              { name: 'CU_ENDPOINT', value: contentUnderstandingEndpoint }
              { name: 'CU_API_VERSION', value: '2025-11-01' }
              { name: 'CU_ANALYZER_ID', value: 'prebuilt-documentSearch' }
              { name: 'MAX_WORKERS', value: '8' }
              { name: 'PAGE_CONCURRENCY', value: '2' }
            ]
          )
          volumeMounts: [
            {
              volumeName: 'demo-files'
              mountPath: '/data'
            }
          ]
          probes: [
            {
              type: 'Readiness'
              httpGet: {
                path: effectiveProbePath
                port: effectivePort
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 6
            }
            {
              type: 'Liveness'
              httpGet: {
                path: effectiveProbePath
                port: effectivePort
              }
              initialDelaySeconds: 30
              periodSeconds: 30
              timeoutSeconds: 5
              failureThreshold: 3
            }
          ]
        }
      ]
      volumes: [
        {
          name: 'demo-files'
          storageType: 'AzureFile'
          storageName: storageName
          mountOptions: 'uid=10001,gid=10001,file_mode=0600,dir_mode=0700,nobrl'
        }
      ]
    }
  }
}

output resourceId string = app.id
output fqdn string = app.properties.configuration.ingress.fqdn
