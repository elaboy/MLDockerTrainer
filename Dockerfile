#Build Stage
FROM mcr.microsoft.com/dotnet/sdk:6.0-focal AS build
WORKDIR /App
#Copy everything
COPY . .
#Restore 
RUN dotnet restore 
RUN dotnet publish -c Release -o out 

#Build runtime image
FROM mcr.microsoft.com/dotnet/aspnet:6.0
WORKDIR /App
COPY --from=build /App/out .
ENTRYPOINT ["dotnet", "MLDockerTrainer.dll"]