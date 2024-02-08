#Build Stage
FROM mcr.microsoft.com/dotnet/sdk:6.0-alpine AS build
WORKDIR /App
#Copy everything
COPY . .
#Restore 
RUN dotnet nuget add source .
#Copy nuget packages
COPY *.nupkg /root/.nuget/packages
RUN dotnet restore --verbosity normal
RUN dotnet publish -c Release -o out 

#Build runtime image
FROM mcr.microsoft.com/dotnet/runtime:6.0 As runtime
WORKDIR /App
COPY --from=build /App/out .
ENTRYPOINT ["dotnet", "MLDockerTrainer.dll", "settings.txt"]