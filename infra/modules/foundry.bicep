param accountName string
param location string
param tags object

resource account 'Microsoft.CognitiveServices/accounts@2026-09-01' = {
  name: accountName
  location: location
  tags: tags
  kind: 'AIServices'
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: accountName
    disableLocalAuth: true
    publicNetworkAccess: 'Enabled'
  }
}

resource gptDeployment 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: account
  name: 'gpt-6.1-sol-c280'
  tags: tags
  sku: {
    name: 'GlobalStandard'
    capacity: 200
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-6.1-sol'
      version: '2026-09-29'
    }
  }
}

resource cuMiniDeployment 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: account
  dependsOn: [
    gptDeployment
  ]
  name: 'cu-mini-c280'
  tags: tags
  sku: {
    name: 'GlobalStandard'
    capacity: 50
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4.1-mini'
      version: '2025-04-14'
    }
  }
}

resource cuEmbeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: account
  dependsOn: [
    cuMiniDeployment
  ]
  name: 'cu-embedding-c280'
  tags: tags
  sku: {
    name: 'Standard'
    capacity: 10
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-3-large'
      version: '1'
    }
  }
}

output resourceId string = account.id
output systemPrincipalId string = account.identity.principalId
output openAiEndpoint string = 'https://${account.name}.services.ai.azure.com/openai/v1/'
output contentUnderstandingEndpoint string = 'https://${account.name}.services.ai.azure.com/'
