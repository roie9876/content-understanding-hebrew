param environmentName string
param location string
param tags object
param workspaceCustomerId string
param workspaceName string
param storageAccountName string
param fileShareName string

resource workspace 'Microsoft.OperationalInsights/workspaces@2026-03-01' existing = {
  name: workspaceName
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2026-09-01' existing = {
  name: storageAccountName
}

resource environment 'Microsoft.App/managedEnvironments@2026-07-01' = {
  name: environmentName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: workspaceCustomerId
        sharedKey: workspace.listKeys().primarySharedKey
      }
    }
  }
}

resource environmentStorage 'Microsoft.App/managedEnvironments/storages@2025-01-01' = {
  parent: environment
  name: fileShareName
  properties: {
    azureFile: {
      accountKey: storageAccount.listKeys().keys[0].value
      accountName: storageAccountName
      accessMode: 'ReadWrite'
      shareName: fileShareName
    }
  }
}

output resourceId string = environment.id
output storageName string = environmentStorage.name
