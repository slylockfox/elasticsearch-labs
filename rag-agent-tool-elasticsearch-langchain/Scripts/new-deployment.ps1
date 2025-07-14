# First run:
#    Connect-AzAccount

$model = New-Object -TypeName 'Microsoft.Azure.Management.CognitiveServices.Models.DeploymentModel' -Property @{
    Name = 'gpt-4o'
    Version = '2024-11-20'
    Format = 'OpenAI'
}

$properties = New-Object -TypeName 'Microsoft.Azure.Management.CognitiveServices.Models.DeploymentProperties' -Property @{
    Model = $model
}

$sku = New-Object -TypeName "Microsoft.Azure.Management.CognitiveServices.Models.Sku" -Property @{
    Name = 'Standard'
    Capacity = '1'
}

New-AzCognitiveServicesAccountDeployment `
    -ResourceGroupName "RAG-tutorial" `
    -AccountName "Snyder-OpenAI" `
    -Name "Snyder-OpenAI-deployment" `
    -Properties $properties -Sku $sku
