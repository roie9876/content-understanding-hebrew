targetScope = 'subscription'

@minLength(1)
@maxLength(64)
param environmentName string

@minLength(1)
param location string

param sessionId string
param deployedBy string
param createdAt string
// Reserved for deploy-phase caller context; this plan has no deployer-scoped RBAC target.
#disable-next-line no-unused-params
param deployerObjectId string

@description('Phase 1 uses the public hello-world image. Phase 2 supplies the private ACR image.')
param containerImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

@description('Public demo expiry. The application enforces this value at runtime.')
param demoExpiresAt string = dateTimeAdd(utcNow(), 'P7D')

var tags = {
  'app-onboard-skill': 'true'
  'app-onboard-session-id': sessionId
  'created-at': createdAt
  environment: environmentName
  'deployed-by': deployedBy
}

resource rg 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: 'rg-inksight-dev-c280'
  location: location
  tags: tags
}

module identity './modules/managed-identity.bicep' = {
  name: 'managed-identity'
  scope: rg
  params: {
    identityName: 'mi-inksight-dev-c280'
    location: location
    tags: tags
  }
}

module logs './modules/log-analytics.bicep' = {
  name: 'log-analytics'
  scope: rg
  params: {
    workspaceName: 'log-inksight-dev-c280'
    location: location
    tags: tags
  }
}

module storage './modules/storage.bicep' = {
  name: 'storage'
  scope: rg
  params: {
    storageAccountName: 'stinksightdevc280'
    fileShareName: 'documents-c280'
    location: location
    tags: tags
  }
}

module acr './modules/container-registry.bicep' = {
  name: 'container-registry'
  scope: rg
  params: {
    registryName: 'crinksightdevc280'
    location: location
    tags: tags
  }
}

module foundry './modules/foundry.bicep' = {
  name: 'foundry'
  scope: rg
  params: {
    accountName: 'ai-inksight-dev-c280'
    location: location
    tags: tags
  }
}

module environment './modules/managed-environment.bicep' = {
  name: 'managed-environment'
  scope: rg
  params: {
    environmentName: 'env-inksight-dev-c280'
    location: location
    tags: tags
    workspaceCustomerId: logs.outputs.customerId
    workspaceName: 'log-inksight-dev-c280'
    storageAccountName: storage.outputs.storageAccountName
    fileShareName: storage.outputs.fileShareName
  }
}

module roles './modules/role-assignments.bicep' = {
  name: 'role-assignments'
  scope: rg
  dependsOn: [
    acr
  ]
  params: {
    acrName: 'crinksightdevc280'
    accountName: 'ai-inksight-dev-c280'
    identityPrincipalId: identity.outputs.principalId
    foundryPrincipalId: foundry.outputs.systemPrincipalId
  }
}

module app './modules/container-app.bicep' = {
  name: 'container-app'
  scope: rg
  dependsOn: [
    acr
    environment
    roles
    storage
  ]
  params: {
    appName: 'ca-inksight-dev-c280'
    environmentName: 'env-inksight-dev-c280'
    containerImage: containerImage
    location: location
    tags: tags
    acrName: 'crinksightdevc280'
    managedIdentityResourceId: identity.outputs.resourceId
    managedIdentityClientId: identity.outputs.clientId
    foundryOpenAiEndpoint: foundry.outputs.openAiEndpoint
    contentUnderstandingEndpoint: foundry.outputs.contentUnderstandingEndpoint
    storageName: 'documents-c280'
    demoExpiresAt: demoExpiresAt
  }
}
