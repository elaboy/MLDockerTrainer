#Build Stage
FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /App
#Copy everything
COPY . .

RUN dotnet nuget add source /root/.nuget/packages/
RUN dotnet restore --verbosity normal 
RUN dotnet publish -c Release -o /App/out 

#Build runtime image
FROM mcr.microsoft.com/dotnet/runtime:6.0 As runtime
WORKDIR /App
COPY --from=build /App/out .
ENTRYPOINT ["dotnet", "MLDockerTrainer.dll", "settings.txt"]