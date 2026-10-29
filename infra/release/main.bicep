targetScope = 'subscription'

param resourceGroupName string
param appName string
param environmentName string
param registryName string
param identityName string
param identityClientId string
param foundryAccountName string
param storageName string
param containerImage string
param location string
param appTags object

resource rg 'Microsoft.Resources/resourceGroups@2023-07-01' existing = {
  name: resourceGroupName
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2024-11-30' existing = {
  scope: rg
  name: identityName
}

resource registry 'Microsoft.ContainerRegistry/registries@2025-11-01' existing = {
  scope: rg
  name: registryName
}

module app '../modules/container-app.bicep' = {
  name: 'inksight-release'
  scope: rg
  params: {
    appName: appName
    environmentName: environmentName
    containerImage: containerImage
    location: location
    tags: appTags
    acrName: registryName
    registryLoginServer: '${registryName}.azurecr.io'
    managedIdentityResourceId: identity.id
    managedIdentityClientId: identityClientId
    foundryOpenAiEndpoint: 'https://${foundryAccountName}.services.ai.azure.com/openai/v1/'
    contentUnderstandingEndpoint: 'https://${foundryAccountName}.services.ai.azure.com/'
    storageName: storageName
    demoExpiresAt: ''
  }
}

output AZURE_RESOURCE_GROUP string = rg.name
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = registry.properties.loginServer
output AZURE_CONTAINER_APP_NAME string = appName
output SERVICE_INKSIGHT_RESOURCE_ID string = app.outputs.resourceId
output SERVICE_INKSIGHT_ENDPOINT_URL string = 'https://${app.outputs.fqdn}'